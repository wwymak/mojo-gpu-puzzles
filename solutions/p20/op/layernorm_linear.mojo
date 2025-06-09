from math import sqrt
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from layout.tensor_builder import LayoutTensorBuild as tb
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

alias TPB = 16
alias dtype = DType.float32


# Idiomatic tiled matmul from p14.mojo - adapted for [batch*seq, hidden] @ [hidden, output] -> [batch*seq, output]
fn matmul_idiomatic_tiled[
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    rows: Int,
    cols: Int,
    inner_dim: Int,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, a_layout],
    b: LayoutTensor[mut=False, dtype, b_layout],
):
    """Idiomatic tiled matmul following p14.mojo exactly."""
    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tile[TPB, TPB](block_idx.x, block_idx.y)
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    local_row = thread_idx.x
    local_col = thread_idx.y

    var acc: output.element_type = 0

    alias load_a_layout = Layout.row_major(1, TPB)
    alias load_b_layout = Layout.row_major(TPB, 1)

    for idx in range((inner_dim + TPB - 1) // TPB):
        a_tile = a.tile[TPB, TPB](block_idx.x, idx)
        b_tile = b.tile[TPB, TPB](idx, block_idx.y)

        copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)

        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    if (
        block_idx.x * TPB + local_row < rows
        and block_idx.y * TPB + local_col < cols
    ):
        out_tile[local_row, local_col] = acc


# Simple LayerNorm kernel
fn layernorm_kernel[
    input_layout: Layout,
    ln_params_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
](
    output: LayoutTensor[mut=True, dtype, output_layout],
    input: LayoutTensor[mut=False, dtype, input_layout],
    ln_weight: LayoutTensor[mut=False, dtype, ln_params_layout],
    ln_bias: LayoutTensor[mut=False, dtype, ln_params_layout],
):
    """Simple LayerNorm kernel like p14.mojo naive."""
    batch_idx = block_idx.x
    seq_idx = block_idx.y
    hidden_idx = thread_idx.x

    if (
        batch_idx >= batch_size
        or seq_idx >= seq_len
        or hidden_idx >= hidden_dim
    ):
        return

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        var val = input[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    var mean_val = sum_val / hidden_dim
    var var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    var inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Apply LayerNorm to this element
    var input_val = input[batch_idx, seq_idx, hidden_idx]
    var normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
        ln_weight[hidden_idx]
    ) + rebind[Scalar[dtype]](ln_bias[hidden_idx])
    output[batch_idx, seq_idx, hidden_idx] = normalized


# Optimized transpose kernel from p17
fn transpose_kernel[
    layout_in: Layout,
    layout_out: Layout,
    rows: Int,
    cols: Int,
](
    output: LayoutTensor[mut=True, dtype, layout_out],
    input: LayoutTensor[mut=False, dtype, layout_in],
):
    """Transpose matrix using shared memory tiling for coalesced access."""
    shared_tile = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    local_row = thread_idx.y
    local_col = thread_idx.x

    global_row = block_idx.y * TPB + local_row
    global_col = block_idx.x * TPB + local_col

    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = input[global_row, global_col]
    else:
        shared_tile[local_row, local_col] = 0.0

    barrier()

    out_row = block_idx.x * TPB + local_row
    out_col = block_idx.y * TPB + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]


# Simple bias addition kernel
fn add_bias_kernel[
    input_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    output_dim: Int,
](
    output: LayoutTensor[mut=True, dtype, output_layout],
    input: LayoutTensor[mut=False, dtype, input_layout],
    bias: LayoutTensor[mut=False, dtype, bias_layout],
):
    """Simple bias addition."""
    batch_idx = block_idx.x
    seq_idx = block_idx.y
    out_idx = thread_idx.x

    if batch_idx >= batch_size or seq_idx >= seq_len or out_idx >= output_dim:
        return

    output[batch_idx, seq_idx, out_idx] = input[
        batch_idx, seq_idx, out_idx
    ] + rebind[Scalar[dtype]](bias[out_idx])


fn minimal_fused_kernel[
    input_layout: Layout,
    ln_params_layout: Layout,
    weight_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
](
    output: LayoutTensor[mut=True, dtype, output_layout],
    input: LayoutTensor[mut=False, dtype, input_layout],
    ln_weight: LayoutTensor[mut=False, dtype, ln_params_layout],
    ln_bias: LayoutTensor[mut=False, dtype, ln_params_layout],
    linear_weight: LayoutTensor[mut=False, dtype, weight_layout],
    linear_bias: LayoutTensor[mut=False, dtype, bias_layout],
):
    """Minimal fused kernel - one thread per sequence position to avoid redundancy.
    """
    # Grid: (batch_size, seq_len) - one thread block per sequence position
    # Block: (1,) - single thread per sequence position to avoid redundant computation
    batch_idx = block_idx.x
    seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Step 1: Compute LayerNorm statistics once per sequence position
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        var val = input[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    var mean_val = sum_val / hidden_dim
    var var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    var inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Step 2: Compute all outputs for this sequence position
    @parameter
    for out_idx in range(output_dim):
        var acc: Scalar[dtype] = 0

        @parameter
        for h in range(hidden_dim):
            var input_val = input[batch_idx, seq_idx, h]
            var normalized = (input_val - mean_val) * inv_std * rebind[
                Scalar[dtype]
            ](ln_weight[h]) + rebind[Scalar[dtype]](ln_bias[h])
            acc += rebind[Scalar[dtype]](normalized * linear_weight[out_idx, h])

        output[batch_idx, seq_idx, out_idx] = acc + rebind[Scalar[dtype]](
            linear_bias[out_idx]
        )


@compiler.register("layernorm_linear")
struct LayerNormLinearCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        algorithm: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
    ](
        output: OutputTensor[dtype = DType.float32, rank=3],
        input: InputTensor[dtype = DType.float32, rank=3],
        ln_weight: InputTensor[dtype = DType.float32, rank=1],
        ln_bias: InputTensor[dtype = DType.float32, rank=1],
        linear_weight: InputTensor[dtype = DType.float32, rank=2],
        linear_bias: InputTensor[dtype = DType.float32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        output_tensor = output.to_layout_tensor()
        input_tensor = input.to_layout_tensor()
        ln_weight_tensor = ln_weight.to_layout_tensor()
        ln_bias_tensor = ln_bias.to_layout_tensor()
        linear_weight_tensor = linear_weight.to_layout_tensor()
        linear_bias_tensor = linear_bias.to_layout_tensor()

        alias input_layout = input_tensor.layout
        alias ln_params_layout = ln_weight_tensor.layout
        alias weight_layout = linear_weight_tensor.layout
        alias bias_layout = linear_bias_tensor.layout
        alias output_layout = output_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            @parameter
            if algorithm == "fused":
                # fused case - one thread per sequence position
                gpu_ctx.enqueue_function[
                    minimal_fused_kernel[
                        input_layout,
                        ln_params_layout,
                        weight_layout,
                        bias_layout,
                        output_layout,
                        batch_size,
                        seq_len,
                        hidden_dim,
                        output_dim,
                    ]
                ](
                    output_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    linear_weight_tensor,
                    linear_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(1,),
                )
            elif algorithm == "unfused":
                # unfused case
                # Create intermediate normalized tensor
                var normalized_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * hidden_dim
                )
                var normalized_tensor = LayoutTensor[
                    mut=True, dtype, input_layout
                ](normalized_buffer.unsafe_ptr())

                # Step 1: LayerNorm kernel
                gpu_ctx.enqueue_function[
                    layernorm_kernel[
                        input_layout,
                        ln_params_layout,
                        input_layout,
                        batch_size,
                        seq_len,
                        hidden_dim,
                    ]
                ](
                    normalized_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(min(hidden_dim, TPB),),
                )

                # Step 2: Matmul on normalized data
                var total_rows = batch_size * seq_len
                var blocks_x = (total_rows + TPB - 1) // TPB
                var blocks_y = (output_dim + TPB - 1) // TPB

                # Create intermediate result without bias
                var matmul_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * output_dim
                )
                var matmul_tensor = LayoutTensor[
                    mut=True, dtype, output_layout
                ](matmul_buffer.unsafe_ptr())

                # Create transposed weight matrix: [output_dim, hidden_dim] -> [hidden_dim, output_dim]
                var transposed_weight_buffer = gpu_ctx.enqueue_create_buffer[
                    dtype
                ](hidden_dim * output_dim)
                var transposed_weight_tensor = LayoutTensor[
                    mut=True, dtype, Layout.row_major(hidden_dim, output_dim)
                ](transposed_weight_buffer.unsafe_ptr())

                # Transpose the weight matrix
                var transpose_blocks_x = (hidden_dim + TPB - 1) // TPB
                var transpose_blocks_y = (output_dim + TPB - 1) // TPB
                gpu_ctx.enqueue_function[
                    transpose_kernel[
                        weight_layout,
                        transposed_weight_tensor.layout,
                        output_dim,
                        hidden_dim,
                    ]
                ](
                    transposed_weight_tensor,
                    linear_weight_tensor,
                    grid_dim=(transpose_blocks_x, transpose_blocks_y),
                    block_dim=(TPB, TPB),
                )

                # Reshape tensors for matmul: [batch*seq, hidden] @ [hidden, output] -> [batch*seq, output]
                var flat_normalized = normalized_tensor.reshape[
                    Layout.row_major(batch_size * seq_len, hidden_dim)
                ]()
                var flat_matmul = matmul_tensor.reshape[
                    Layout.row_major(batch_size * seq_len, output_dim)
                ]()

                gpu_ctx.enqueue_function[
                    matmul_idiomatic_tiled[
                        flat_normalized.layout,
                        transposed_weight_tensor.layout,
                        flat_matmul.layout,
                        batch_size * seq_len,
                        output_dim,
                        hidden_dim,
                    ]
                ](
                    flat_matmul,
                    flat_normalized,
                    transposed_weight_tensor,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(TPB, TPB),
                )

                # Step 3: Add bias - reshape matmul result back to 3D for bias addition
                var reshaped_matmul = matmul_tensor.reshape[
                    Layout.row_major(batch_size, seq_len, output_dim)
                ]()

                gpu_ctx.enqueue_function[
                    add_bias_kernel[
                        reshaped_matmul.layout,
                        bias_layout,
                        output_layout,
                        batch_size,
                        seq_len,
                        output_dim,
                    ]
                ](
                    output_tensor,
                    reshaped_matmul,
                    linear_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(min(output_dim, TPB),),
                )

        elif target == "cpu":
            # CPU implementation - always fused (no separate kernels for CPU)
            # Note: CPU doesn't have separate fused vs unfused - both use the same implementation
            for batch in range(batch_size):
                for seq in range(seq_len):
                    # LayerNorm
                    var sum_val: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        sum_val += rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                    var mean_val = sum_val / hidden_dim

                    var var_sum: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        var diff = input_tensor[batch, seq, h] - mean_val
                        var_sum += rebind[Scalar[dtype]](diff * diff)
                    var var_val = var_sum / hidden_dim
                    var inv_std = 1.0 / sqrt(var_val + 1e-5)

                    # Apply LayerNorm and Linear in one step (truly fused)
                    for out_idx in range(output_dim):
                        var acc: Scalar[dtype] = 0
                        for h in range(hidden_dim):
                            var input_val = input_tensor[batch, seq, h]
                            var normalized = (
                                input_val - mean_val
                            ) * inv_std * ln_weight_tensor[h] + ln_bias_tensor[
                                h
                            ]
                            acc += rebind[Scalar[dtype]](
                                normalized * linear_weight_tensor[out_idx, h]
                            )
                        output_tensor[batch, seq, out_idx] = (
                            acc + linear_bias_tensor[out_idx]
                        )

        else:
            raise Error("Unsupported target: " + target)
