from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from sys import argv

alias SIZE = 4
alias MATRIX_SIZE = 3
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32
alias vector_layout = Layout.row_major(SIZE)
alias ITER = 2


# ANCHOR: first_crash
fn add_10(
    result: UnsafePointer[Scalar[dtype]], input: UnsafePointer[Scalar[dtype]]
):
    shared = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    barrier()
    if global_i < size :
        shared[local_i] = a[global_i]
        if global_i == 0:
            output[global_i] = shared[local_i]
        elif global_i == 1:
            output[global_i] = shared[local_i] + shared[local_i -1 ]
        else:
            output[global_i] = shared[local_i] + shared[local_i -1 ]  + shared[local_i -2 ]

    # FILL ME IN (roughly 10 lines)


# ANCHOR_END: first_crash


# ANCHOR: second_crash
fn process_sliding_window(
    output: LayoutTensor[mut=True, dtype, vector_layout],
    input: LayoutTensor[mut=False, dtype, vector_layout],
):
    thread_id = thread_idx.x

    # Each thread processes a sliding window of 3 elements
    window_sum = Scalar[dtype](0.0)

    # Sum elements in sliding window: [i-1, i, i+1]
    for offset in range(ITER):
        idx = thread_id + offset - 1
        if 0 <= idx < SIZE:
            value = rebind[Scalar[dtype]](input[idx])
            window_sum += value

    output[thread_id] = window_sum


# ANCHOR_END: second_crash


# ANCHOR: third_crash
fn collaborative_filter(
    output: LayoutTensor[mut=True, dtype, vector_layout],
    input: LayoutTensor[mut=False, dtype, vector_layout],
):
    thread_id = thread_idx.x

    # Shared memory workspace for collaborative processing
    shared_workspace = tb[dtype]().row_major[SIZE - 1]().shared().alloc()

    # Phase 1: Initialize shared workspace (all threads participate)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] = rebind[Scalar[dtype]](input[thread_id])
    barrier()

    # Phase 2: Collaborative processing
    if thread_id < SIZE - 1:
        # Apply collaborative filter with neighbors
        if thread_id > 0:
            shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
        barrier()

    # Phase 3: Final synchronization and output
    barrier()

    # Write filtered results back to output
    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = rebind[Scalar[dtype]](input[thread_id])


# ANCHOR_END: third_crash


def main():
    if len(argv()) != 2:
        print(
            "Usage: pixi run mojo p09 [--first-case | --second-case |"
            " --third-case]"
        )
        return

    if argv()[1] == "--first-case":
        print(
            "First Case: Try to identify what's wrong without looking at the"
            " code!"
        )
        print()

        with DeviceContext() as ctx:
            input_ptr = UnsafePointer[Scalar[dtype]]()
            result_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            # Enqueue function
            ctx.enqueue_function[add_10](
                result_buf.unsafe_ptr(),
                input_ptr,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with result_buf.map_to_host() as result_host:
                print("result:", result_host)

    elif argv()[1] == "--second-case":
        print("This program computes sliding window sums for each position...")
        print()

        with DeviceContext() as ctx:
            # Create buffers
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            # Initialize input [0, 1, 2, 3]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = i

            # Create LayoutTensors for structured access
            input_tensor = LayoutTensor[mut=False, dtype, vector_layout](
                input_buf.unsafe_ptr()
            )
            output_tensor = LayoutTensor[mut=True, dtype, vector_layout](
                output_buf.unsafe_ptr()
            )

            print("Input array: [0, 1, 2, 3]")
            print("Computing sliding window sums (window size = 3)...")
            print(
                "Each position should sum its neighbors: [left + center +"
                " right]"
            )

            ctx.enqueue_function[process_sliding_window](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with output_buf.map_to_host() as output_host:
                print("Actual result:", output_host)

                # Expected sliding window results
                expected_0 = Scalar[dtype](1.0)
                expected_1 = Scalar[dtype](3.0)
                expected_2 = Scalar[dtype](6.0)
                expected_3 = Scalar[dtype](5.0)
                print("Expected: [1.0, 3.0, 6.0, 5.0]")

                # Check if results match expected pattern
                matches = True
                if abs(output_host[0] - expected_0) > 0.001:
                    matches = False
                if abs(output_host[1] - expected_1) > 0.001:
                    matches = False
                if abs(output_host[2] - expected_2) > 0.001:
                    matches = False
                if abs(output_host[3] - expected_3) > 0.001:
                    matches = False

                if matches:
                    print(
                        "[PASS] Test PASSED - Sliding window sums are correct"
                    )
                else:
                    print(
                        "[FAIL] Test FAILED - Sliding window sums are"
                        " incorrect!"
                    )
                    print("Check the window indexing logic...")

    elif argv()[1] == "--third-case":
        print(
            "Third Case: Advanced collaborative filtering with shared memory..."
        )
        print("WARNING: This may hang - use Ctrl+C to stop if needed")
        print()

        with DeviceContext() as ctx:
            # Create input and output buffers
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

            # Initialize input data [1, 2, 3, 4]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = i + 1

            # Create LayoutTensors
            input_tensor = LayoutTensor[mut=False, dtype, vector_layout](
                input_buf.unsafe_ptr()
            )
            output_tensor = LayoutTensor[mut=True, dtype, vector_layout](
                output_buf.unsafe_ptr()
            )

            print("Input array: [1, 2, 3, 4]")
            print("Applying collaborative filter using shared memory...")
            print("Each thread cooperates with neighbors for smoothing...")

            # This will likely hang due to barrier deadlock
            ctx.enqueue_function[collaborative_filter](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            print("Waiting for GPU computation to complete...")
            ctx.synchronize()

            with output_buf.map_to_host() as output_host:
                print("Result:", output_host)
                print(
                    "[SUCCESS] Collaborative filtering completed successfully!"
                )

    else:
        print(
            "Unsupported option. Choose between [--first-case, --second-case,"
            " --third-case]"
        )
