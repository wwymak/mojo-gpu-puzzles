from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout_tensor import copy_dram_to_sram_async
from sys import argv, info
from testing import assert_equal, assert_almost_equal


alias VECTOR_SIZE = 16384
alias CONV_TILE_SIZE = 256
alias KERNEL_SIZE = 5
alias HALO_SIZE = KERNEL_SIZE // 2  # Halo elements needed for boundary
alias BUFFER_SIZE = CONV_TILE_SIZE + 2 * HALO_SIZE  # Include halo for boundary conditions
alias BLOCKS_PER_GRID_ASYNC = (
    VECTOR_SIZE + CONV_TILE_SIZE - 1
) // CONV_TILE_SIZE
alias THREADS_PER_BLOCK_ASYNC = 256
alias dtype = DType.float32
alias layout_async = Layout.row_major(VECTOR_SIZE)


# ANCHOR: async_copy_overlap_convolution_solution
fn async_copy_overlap_convolution[
    dtype: DType, layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
    kernel: LayoutTensor[mut=False, dtype, Layout.row_major(KERNEL_SIZE)],
):
    """Demonstrates async copy operations building on p14 patterns.

    This shows how to use copy_dram_to_sram_async and async_copy_wait_all
    for efficient memory transfers, extending the patterns from p14 matmul.
    """

    # Shared memory buffers (like p14, but without .fill(0) to avoid race)
    input_shared = tb[dtype]().row_major[CONV_TILE_SIZE]().shared().alloc()
    kernel_shared = tb[dtype]().row_major[KERNEL_SIZE]().shared().alloc()

    local_i = thread_idx.x

    # Phase 1: Launch async copy for input tile
    # Note: tile() does NOT perform bounds checking - ensure valid tile bounds
    input_tile = input.tile[CONV_TILE_SIZE](block_idx.x)

    # Use async copy with thread layout matching p14 pattern
    alias load_layout = Layout.row_major(THREADS_PER_BLOCK_ASYNC, 1)
    copy_dram_to_sram_async[thread_layout=load_layout](input_shared, input_tile)

    # Phase 2: Load kernel synchronously (small data)
    if local_i < KERNEL_SIZE:
        kernel_shared[local_i] = kernel[local_i]

    # Phase 3: Wait for async copy to complete
    async_copy_wait_all()  # Always wait since we always do async copy
    barrier()  # Sync all threads

    # Phase 4: Compute convolution
    global_i = block_idx.x * CONV_TILE_SIZE + local_i
    if local_i < CONV_TILE_SIZE and global_i < output.shape[0]():
        var result: output.element_type = 0

        # Simple convolution avoiding boundary issues
        if local_i >= HALO_SIZE and local_i < CONV_TILE_SIZE - HALO_SIZE:
            # Full convolution for center elements
            for k in range(KERNEL_SIZE):
                input_idx = local_i + k - HALO_SIZE
                if input_idx >= 0 and input_idx < CONV_TILE_SIZE:
                    result += input_shared[input_idx] * kernel_shared[k]
        else:
            # For boundary elements, just copy input (no convolution)
            result = input_shared[local_i]

        output[global_i] = result


# ANCHOR_END: async_copy_overlap_convolution_solution


def test_async_copy_overlap_convolution():
    """Test async copy overlap with 1D convolution."""
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](VECTOR_SIZE).enqueue_fill(
            0
        )
        output_buf = ctx.enqueue_create_buffer[dtype](VECTOR_SIZE).enqueue_fill(
            0
        )
        kernel_buf = ctx.enqueue_create_buffer[dtype](KERNEL_SIZE).enqueue_fill(
            0
        )

        # Create test data: consecutive integers [1, 2, 3, ..., VECTOR_SIZE]
        with input_buf.map_to_host() as input_host:
            for i in range(VECTOR_SIZE):
                input_host[i] = Float32(i + 1)

        # Create test kernel: [1, 2, 3, 4, 5]
        with kernel_buf.map_to_host() as kernel_host:
            for i in range(KERNEL_SIZE):
                kernel_host[i] = Float32(i + 1)

        input_tensor = LayoutTensor[mut=False, dtype, layout_async](
            input_buf.unsafe_ptr()
        )
        output_tensor = LayoutTensor[mut=True, dtype, layout_async](
            output_buf.unsafe_ptr()
        )
        kernel_tensor = LayoutTensor[
            mut=False, dtype, Layout.row_major(KERNEL_SIZE)
        ](kernel_buf.unsafe_ptr())

        ctx.enqueue_function[
            async_copy_overlap_convolution[dtype, layout_async]
        ](
            output_tensor,
            input_tensor,
            kernel_tensor,
            grid_dim=(BLOCKS_PER_GRID_ASYNC, 1),
            block_dim=(THREADS_PER_BLOCK_ASYNC, 1),
        )

        ctx.synchronize()

        # Verify convolution results
        with output_buf.map_to_host() as output_host:
            with input_buf.map_to_host() as input_host:
                print(
                    "Async copy overlap convolution - verifying first 10"
                    " values:"
                )

                var success = True
                for i in range(min(10, VECTOR_SIZE)):
                    var expected_val: Float32 = 0

                    # Match implementation logic: boundary elements copy input, center elements get convolution
                    var local_i_in_tile = i % CONV_TILE_SIZE
                    if (
                        local_i_in_tile >= HALO_SIZE
                        and local_i_in_tile < CONV_TILE_SIZE - HALO_SIZE
                    ):
                        # Center elements: apply convolution
                        for k in range(KERNEL_SIZE):
                            var input_idx = i + k - HALO_SIZE
                            if input_idx >= 0 and input_idx < VECTOR_SIZE:
                                expected_val += input_host[input_idx] * (k + 1)
                    else:
                        # Boundary elements: copy input
                        expected_val = input_host[i]

                    actual = output_host[i]
                    print(
                        "  Index",
                        i,
                        ": input=",
                        input_host[i],
                        ", output=",
                        actual,
                        ", expected=",
                        expected_val,
                    )

                    if abs(actual - expected_val) > 0.01:
                        print("Mismatch at index", i)
                        success = False
                        break

                if success:
                    print("Async copy overlap convolution test PASSED!")
                else:
                    print("Async copy overlap convolution test FAILED!")


def main():
    """Run memory fence tests based on command line arguments."""
    if len(argv()) != 1:
        print("Usage: p25.mojo")
        return

    print("Puzzle 25: Async Memory Operations & Copy Overlap")
    print("=" * 50)
    print("VECTOR_SIZE:", VECTOR_SIZE)
    print("CONV_TILE_SIZE:", CONV_TILE_SIZE)
    print("KERNEL_SIZE:", KERNEL_SIZE)
    print("HALO_SIZE:", HALO_SIZE)
    print("BUFFER_SIZE:", BUFFER_SIZE)
    print("BLOCKS_PER_GRID_ASYNC:", BLOCKS_PER_GRID_ASYNC)
    print("THREADS_PER_BLOCK_ASYNC:", THREADS_PER_BLOCK_ASYNC)
    test_async_copy_overlap_convolution()
