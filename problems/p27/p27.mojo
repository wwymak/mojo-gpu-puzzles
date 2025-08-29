from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from os.atomic import Atomic
from gpu.warp import WARP_SIZE
from gpu import block
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import argv
from testing import assert_equal
from math import floor


# ANCHOR: block_sum_dot_product
alias SIZE = 128
alias TPB = 128
alias NUM_BINS = 8
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)
alias dtype = DType.float32


fn block_sum_dot_product[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    """Dot product using block.sum() - convenience function like warp.sum()!
    Replaces manual shared memory + barriers + tree reduction with one line."""

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # FILL IN (roughly 6 lines)


# ANCHOR_END: block_sum_dot_product

# ANCHOR: block_histogram
alias bin_layout = Layout.row_major(SIZE)  # Max SIZE elements per bin


fn block_histogram_bin_extract[
    in_layout: Layout, bin_layout: Layout, out_layout: Layout, tpb: Int
](
    input_data: LayoutTensor[mut=False, dtype, in_layout],
    bin_output: LayoutTensor[mut=True, dtype, bin_layout],
    count_output: LayoutTensor[mut=True, DType.int32, out_layout],
    size: Int,
    target_bin: Int,
    num_bins: Int,
):
    """Parallel histogram using block.prefix_sum() for bin extraction.

    This demonstrates advanced parallel filtering and extraction:
    1. Each thread determines which bin its element belongs to
    2. Use block.prefix_sum() to compute write positions for target_bin elements
    3. Extract and pack only elements belonging to target_bin
    """

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Step 1: Each thread determines its bin and element value

    # FILL IN (roughly 9 lines)

    # Step 2: Create predicate for target bin extraction

    # FILL IN (roughly 3 line)

    # Step 3: Use block.prefix_sum() for parallel bin extraction!
    # This computes where each thread should write within the target bin

    # FILL IN (1 line)

    # Step 4: Extract and pack elements belonging to target_bin

    # FILL IN (roughly 2 line)

    # Step 5: Final thread computes total count for this bin

    # FILL IN (roughly 3 line)


# ANCHOR_END: block_histogram

# ANCHOR: block_normalize

alias vector_layout = Layout.row_major(SIZE)


fn block_normalize_vector[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    input_data: LayoutTensor[mut=False, dtype, in_layout],
    output_data: LayoutTensor[mut=True, dtype, out_layout],
    size: Int,
):
    """Vector mean normalization using block.sum() + block.broadcast() combination.

    This demonstrates the complete block operations workflow:
    1. Use block.sum() to compute sum of all elements (all → one)
    2. Thread 0 computes mean = sum / size
    3. Use block.broadcast() to share mean to all threads (one → all)
    4. Each thread normalizes: output[i] = input[i] / mean
    """

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Step 1: Each thread loads its element

    # FILL IN (roughly 3 lines)

    # Step 2: Use block.sum() to compute total sum (familiar from earlier!)

    # FILL IN (1 line)

    # Step 3: Thread 0 computes mean value

    # FILL IN (roughly 4 lines)

    # Step 4: block.broadcast() shares mean to ALL threads!
    # This completes the block operations trilogy demonstration

    # FILL IN (1 line)

    # Step 5: Each thread normalizes by the mean

    # FILL IN (roughly 3 lines)


# ANCHOR_END: block_normalize


def main():
    if len(argv()) != 2:
        print(
            "Usage: --traditional-dot-product | --block-sum-dot-product |"
            " --histogram | --normalize"
        )
        return

    with DeviceContext() as ctx:
        if argv()[1] == "--traditional-dot-product":
            out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
            b_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            var expected: Scalar[dtype] = 0.0
            with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = i
                    b_host[i] = 2 * i
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = LayoutTensor[mut=False, dtype, in_layout](a.unsafe_ptr())
            b_tensor = LayoutTensor[mut=False, dtype, in_layout](
                b_buf.unsafe_ptr()
            )
            out_tensor = LayoutTensor[mut=True, dtype, out_layout](
                out.unsafe_ptr()
            )

            # Traditional approach: works perfectly when size == TPB
            ctx.enqueue_function[
                traditional_dot_product[in_layout, out_layout, TPB]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                SIZE,
                grid_dim=(1, 1),  # ✅ Single block works when size == TPB
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with out.map_to_host() as result_host:
                result = result_host[0]
                print("Traditional result:", result)
                assert_equal(result, expected)
                print("Complex: shared memory + barriers + tree reduction")

        elif argv()[1] == "--block-sum-dot-product":
            out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
            b_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            var expected: Scalar[dtype] = 0.0
            with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = i
                    b_host[i] = 2 * i
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = LayoutTensor[mut=False, dtype, in_layout](a.unsafe_ptr())
            b_tensor = LayoutTensor[mut=False, dtype, in_layout](
                b_buf.unsafe_ptr()
            )
            out_tensor = LayoutTensor[mut=True, dtype, out_layout](
                out.unsafe_ptr()
            )

            # Block.sum(): Same result with dramatically simpler code!
            ctx.enqueue_function[
                block_sum_dot_product[in_layout, out_layout, TPB]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                SIZE,
                grid_dim=(1, 1),  # Same single block as traditional
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with out.map_to_host() as result_host:
                result = result_host[0]
                print("Block.sum result:", result)
                assert_equal(result, expected)
                print("Block.sum() gives identical results!")
                print(
                    "Compare the code: 15+ lines of barriers → 1 line of"
                    " block.sum()!"
                )
                print("Just like warp.sum() but for the entire block")

        elif argv()[1] == "--histogram":
            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("NUM_BINS:", NUM_BINS)
            print()

            # Create input data with known distribution across bins
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            # Create test data: values distributed across 8 bins [0.0, 1.0)
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    # Create values: 0.1, 0.2, 0.3, ..., cycling through bins
                    input_host[i] = (
                        Float32(i % 80) / 100.0
                    )  # Values [0.0, 0.79]

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print()

            input_tensor = LayoutTensor[mut=False, dtype, in_layout](
                input_buf.unsafe_ptr()
            )

            # Demonstrate histogram for each bin using block.prefix_sum()
            for target_bin in range(NUM_BINS):
                print(
                    "=== Processing Bin",
                    target_bin,
                    "(range [",
                    Float32(target_bin) / NUM_BINS,
                    ",",
                    Float32(target_bin + 1) / NUM_BINS,
                    ")) ===",
                )

                # Create output buffers for this bin
                bin_data = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
                    0
                )
                bin_count = ctx.enqueue_create_buffer[DType.int32](
                    1
                ).enqueue_fill(0)

                bin_tensor = LayoutTensor[mut=True, dtype, bin_layout](
                    bin_data.unsafe_ptr()
                )
                count_tensor = LayoutTensor[mut=True, DType.int32, out_layout](
                    bin_count.unsafe_ptr()
                )

                # Execute histogram kernel for this specific bin
                ctx.enqueue_function[
                    block_histogram_bin_extract[
                        in_layout, bin_layout, out_layout, TPB
                    ]
                ](
                    input_tensor,
                    bin_tensor,
                    count_tensor,
                    SIZE,
                    target_bin,
                    NUM_BINS,
                    grid_dim=(
                        1,
                        1,
                    ),  # Single block demonstrates block.prefix_sum()
                    block_dim=(TPB, 1),
                )

                ctx.synchronize()

                # Display results for this bin
                with bin_count.map_to_host() as count_host:
                    count = count_host[0]
                    print("Bin", target_bin, "count:", count)

                with bin_data.map_to_host() as bin_host:
                    print("Bin", target_bin, "extracted elements:", end=" ")
                    for i in range(min(8, Int(count))):
                        print(bin_host[i], end=" ")
                    if count > 8:
                        print("...")
                    else:
                        print()
                print()

        elif argv()[1] == "--normalize":
            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print()

            # Create input data with known values for easy verification
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            # Create test data: values like [1, 2, 3, 4, 5, ..., 8, 1, 2, 3, ...]
            # Mean value will be 4.5, so normalized values will be input[i] / 4.5
            var sum_value: Scalar[dtype] = 0.0
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    # Create values cycling 1-8, mean will be 4.5
                    value = Float32(
                        (i % 8) + 1
                    )  # Values 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, ...
                    input_host[i] = value
                    sum_value += value

            var mean_value = sum_value / Float32(SIZE)

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print("Sum value:", sum_value)
            print("Mean value:", mean_value)
            print()

            input_tensor = LayoutTensor[mut=False, dtype, in_layout](
                input_buf.unsafe_ptr()
            )
            output_tensor = LayoutTensor[mut=True, dtype, vector_layout](
                output_buf.unsafe_ptr()
            )

            # Execute vector normalization kernel
            ctx.enqueue_function[
                block_normalize_vector[in_layout, vector_layout, TPB]
            ](
                input_tensor,
                output_tensor,
                SIZE,
                grid_dim=(1, 1),  # Single block demonstrates block.broadcast()
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            # Verify results
            print("Mean Normalization Results:")
            with output_buf.map_to_host() as output_host:
                print("Normalized sample:", end=" ")
                for i in range(min(16, SIZE)):
                    print(output_host[i], end=" ")
                print("...")

                # Verify that the mean normalization worked (mean of output should be ~1.0)
                var output_sum: Scalar[dtype] = 0.0
                for i in range(SIZE):
                    output_sum += output_host[i]

                var output_mean = output_sum / Float32(SIZE)
                print("Output sum:", output_sum)
                print("Output mean:", output_mean)
                print(
                    "✅ Success: Output mean is",
                    output_mean,
                    "(should be close to 1.0)",
                )
        else:
            print(
                "Available options: [--traditional-dot-product |"
                " --block-sum-dot-product | --histogram | --normalize]"
            )
