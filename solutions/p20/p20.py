import torch
import time
import argparse
from pathlib import Path

# Test configuration
BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_DIM = 8
OUTPUT_DIM = 16
EPS = 1e-5

print(f"Testing with dimensions: [{BATCH_SIZE}, {SEQ_LEN}, {HIDDEN_DIM}] -> [{BATCH_SIZE}, {SEQ_LEN}, {OUTPUT_DIM}]")

# Cache for compiled operations
_mojo_ops_cache = None

def get_mojo_ops():
    """Get cached CustomOpLibrary instance to avoid recompilation."""
    global _mojo_ops_cache
    if _mojo_ops_cache is None:
        try:
            from max.torch import CustomOpLibrary
            mojo_kernels = Path(__file__).parent / "op"
            _mojo_ops_cache = CustomOpLibrary(mojo_kernels)
            print("✅ Loaded Mojo operations library")
            return _mojo_ops_cache
        except ImportError:
            print("⚠️  max.torch not available")
            return None
    return _mojo_ops_cache

def create_test_data(device='cuda'):
    """Create consistent test data for all tests."""
    torch.manual_seed(42)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)
    ln_weight = torch.ones(HIDDEN_DIM, device=device)
    ln_bias = torch.zeros(HIDDEN_DIM, device=device)
    linear_weight = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=device) * 0.02
    linear_bias = torch.zeros(OUTPUT_DIM, device=device)

    return input_tensor, ln_weight, ln_bias, linear_weight, linear_bias

def reference_layernorm_linear(input, ln_weight, ln_bias, linear_weight, linear_bias, eps=EPS):
    """Reference implementation using standard PyTorch operations."""
    # LayerNorm
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (input - mean) / torch.sqrt(var + eps)
    ln_output = normalized * ln_weight + ln_bias

    # Linear transformation
    output = torch.nn.functional.linear(ln_output, linear_weight, linear_bias)
    return output

def run_mojo_implementation(input, ln_weight, ln_bias, linear_weight, linear_bias, algorithm="fused", target="auto"):
    """Generic function to run Mojo implementations."""
    batch_size, seq_len, hidden_dim = input.shape
    output_dim = linear_weight.shape[0]

    ops = get_mojo_ops()
    if ops is None:
        return None, "Mojo operations library not found"

    # Determine target device
    if target == "auto":
        target = "gpu" if input.device.type == 'cuda' else "cpu"

    # Move to correct device if needed
    if target == "cpu":
        input = input.cpu()
        ln_weight = ln_weight.cpu()
        ln_bias = ln_bias.cpu()
        linear_weight = linear_weight.cpu()
        linear_bias = linear_bias.cpu()

    try:
        output = torch.empty(
            (batch_size, seq_len, output_dim),
            dtype=input.dtype,
            device=input.device
        )

        op = ops.layernorm_linear[{
            "algorithm": algorithm,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        }]

        torch.compile(op)(output, input, ln_weight, ln_bias, linear_weight, linear_bias)
        return output, None

    except Exception as e:
        return None, str(e)

def test_implementation(name, algorithm=None, target="auto", reference_output=None, test_data=None):
    """Generic test function for any implementation."""
    if test_data is None:
        test_data = create_test_data()

    input_tensor, ln_weight, ln_bias, linear_weight, linear_bias = test_data

    print(f"\n🧪 Testing {name}")
    print("-" * (15 + len(name)))

    # Get reference if not provided
    if reference_output is None:
        reference_output = reference_layernorm_linear(input_tensor, ln_weight, ln_bias, linear_weight, linear_bias)

    # Test the implementation
    if algorithm is None:
        # Reference implementation
        output = reference_layernorm_linear(input_tensor, ln_weight, ln_bias, linear_weight, linear_bias)
        success_msg = "✅ Reference PyTorch"
        error_msg = "❌ Reference failed"
    else:
        # Mojo implementation
        output, error = run_mojo_implementation(
            input_tensor, ln_weight, ln_bias, linear_weight, linear_bias,
            algorithm=algorithm, target=target
        )
        success_msg = f"✅ Using Mojo {algorithm} kernel ({target.upper()})"
        error_msg = f"⚠️ {algorithm} kernel failed: {error}"

    if output is not None:
        print(success_msg)

        # Move reference to same device for comparison
        if reference_output.device != output.device:
            reference_output = reference_output.to(output.device)

        diff = torch.max(torch.abs(reference_output - output)).item()
        print(f"   Max difference: {diff:.2e}")

        is_correct = diff < 1e-4
        print(f"   Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        return is_correct, output
    else:
        print(error_msg)
        return False, None

def run_comprehensive_test():
    """Run comprehensive test of all implementations."""
    print("=" * 60)
    print("   Puzzle 20: Fused LayerNorm + Linear Comparison")
    print("   (Reference vs CPU vs GPU Unfused vs GPU Fused)")
    print("=" * 60)

    # Create test data once
    test_data = create_test_data()
    input_tensor = test_data[0]

    # Get reference output
    reference_output = reference_layernorm_linear(*test_data)

    results = {}

    # Test CPU implementation
    results['cpu'], _ = test_implementation(
        "CPU Implementation",
        algorithm="fused",
        target="cpu",
        reference_output=reference_output,
        test_data=test_data
    )

    # Test GPU implementations if CUDA available
    if input_tensor.device.type == 'cuda':
        results['gpu_unfused'], _ = test_implementation(
            "GPU Unfused Implementation",
            algorithm="unfused",
            target="gpu",
            reference_output=reference_output,
            test_data=test_data
        )

        results['gpu_fused'], _ = test_implementation(
            "GPU Fused Implementation",
            algorithm="fused",
            target="gpu",
            reference_output=reference_output,
            test_data=test_data
        )
    else:
        print("\n⚠️  CUDA not available - skipping GPU tests")
        results['gpu_unfused'] = False
        results['gpu_fused'] = False

    # Summary
    print(f"\n📊 Summary:")
    print(f"   - CPU:         {'✅ CORRECT' if results['cpu'] else '❌ INCORRECT'}")
    print(f"   - GPU unfused: {'✅ CORRECT' if results['gpu_unfused'] else '❌ INCORRECT'}")
    print(f"   - GPU fused:   {'✅ CORRECT' if results['gpu_fused'] else '❌ INCORRECT'}")

    all_correct = all(results.values())
    print(f"\n   Overall: {'✅ ALL CORRECT' if all_correct else '❌ SOME FAILED'}")

    if all_correct:
        print("\n🎉 Puzzle 20 completed successfully!")
        print("\nWhat we achieved:")
        print("✅ CPU implementation: Fused LayerNorm + Linear")
        print("✅ GPU unfused: Multi-kernel pipeline (LayerNorm → Transpose → Matmul → Bias)")
        print("✅ GPU fused: Single kernel LayerNorm + Linear")
        print("✅ Perfect numerical accuracy (max diff ~1.5e-08)")

        print("\nLearning outcomes:")
        print("- Multi-kernel GPU pipeline design")
        print("- Single-kernel fusion implementation")
        print("- CPU vs GPU optimization strategies")
        print("- Integration of optimized kernels from previous puzzles")
    else:
        print("\n❌ Some implementations failed!")

def run_fused_only_test():
    """Run only the fused kernel test."""
    print("=" * 50)
    print("   Puzzle 20: Testing FUSED KERNEL ONLY")
    print("=" * 50)

    test_data = create_test_data()
    reference_output = reference_layernorm_linear(*test_data)

    is_correct, _ = test_implementation(
        "GPU Fused Implementation ONLY",
        algorithm="fused",
        target="gpu",
        reference_output=reference_output,
        test_data=test_data
    )

    print(f"\n🧪 Fused kernel result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

    if is_correct:
        print("\n🎉 Fused kernel works perfectly!")
    else:
        print("\n❌ Fused kernel failed!")

def benchmark_implementations(algorithm, test_data, iterations=50):
    """Benchmark CPU vs GPU for specific algorithm."""
    print(f"\n⚡ Benchmarking CPU vs GPU {algorithm.upper()}")
    print("-" * (35 + len(algorithm)))

    input_tensor, ln_weight, ln_bias, linear_weight, linear_bias = test_data

    # Skip if not CUDA available
    if input_tensor.device.type != 'cuda':
        print("   ❌ CUDA not available - skipping GPU benchmark")
        return

    times = {}

    # Benchmark CPU
    print("   Testing CPU performance...")
    cpu_output, cpu_error = run_mojo_implementation(*test_data, algorithm="fused", target="cpu")
    if cpu_output is not None:
        # Warmup
        for _ in range(3):
            _ = run_mojo_implementation(*test_data, algorithm="fused", target="cpu")

        start = time.perf_counter()
        for _ in range(iterations):
            _ = run_mojo_implementation(*test_data, algorithm="fused", target="cpu")
        times['cpu'] = time.perf_counter() - start
        print(f"   CPU: {times['cpu']*1000:.2f}ms ({iterations} iterations)")
    else:
        print(f"   CPU failed: {cpu_error}")
        times['cpu'] = None

    # Benchmark GPU
    print(f"   Testing GPU {algorithm} performance...")
    gpu_output, gpu_error = run_mojo_implementation(*test_data, algorithm=algorithm, target="gpu")
    if gpu_output is not None:
        # Warmup
        for _ in range(3):
            _ = run_mojo_implementation(*test_data, algorithm=algorithm, target="gpu")
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = run_mojo_implementation(*test_data, algorithm=algorithm, target="gpu")
        torch.cuda.synchronize()
        times['gpu'] = time.perf_counter() - start
        print(f"   GPU {algorithm}: {times['gpu']*1000:.2f}ms ({iterations} iterations)")
    else:
        print(f"   GPU {algorithm} failed: {gpu_error}")
        times['gpu'] = None

    # Performance comparison
    if times['cpu'] is not None and times['gpu'] is not None:
        speedup = times['cpu'] / times['gpu']
        print(f"\n   📊 GPU {algorithm} vs CPU: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        if speedup > 1:
            print(f"   🚀 GPU {algorithm} wins!")
        else:
            print(f"   🐌 CPU wins (GPU overhead > computation benefit)")
    else:
        print("\n   ❌ Benchmark incomplete due to failures")

def run_algorithm_specific_test(algorithm):
    """Run correctness and benchmark tests for specific algorithm."""
    print("=" * 60)
    print(f"   Puzzle 20: {algorithm.upper()} Algorithm Test & Benchmark")
    print("=" * 60)

    # Create test data
    test_data = create_test_data()
    reference_output = reference_layernorm_linear(*test_data)

    print(f"\n🧪 Correctness Testing for {algorithm.upper()} Algorithm")
    print("=" * (45 + len(algorithm)))

    results = {}

    # Test Reference (for completeness)
    results['reference'], _ = test_implementation(
        "Reference PyTorch Implementation",
        algorithm=None,
        reference_output=reference_output,
        test_data=test_data
    )

    # Test CPU implementation
    results['cpu'], _ = test_implementation(
        "CPU Implementation",
        algorithm="fused",
        target="cpu",
        reference_output=reference_output,
        test_data=test_data
    )

    # Test selected GPU algorithm
    if test_data[0].device.type == 'cuda':
        results[f'gpu_{algorithm}'], _ = test_implementation(
            f"GPU {algorithm.title()} Implementation",
            algorithm=algorithm,
            target="gpu",
            reference_output=reference_output,
            test_data=test_data
        )
    else:
        print(f"\n⚠️  CUDA not available - skipping GPU {algorithm} test")
        results[f'gpu_{algorithm}'] = False

    # Correctness Summary
    print(f"\n📊 Correctness Summary:")
    print(f"   - Reference:   {'✅ CORRECT' if results['reference'] else '❌ INCORRECT'}")
    print(f"   - CPU:         {'✅ CORRECT' if results['cpu'] else '❌ INCORRECT'}")
    print(f"   - GPU {algorithm}: {'✅ CORRECT' if results[f'gpu_{algorithm}'] else '❌ INCORRECT'}")

    all_correct = all(results.values())
    print(f"\n   Overall Correctness: {'✅ ALL CORRECT' if all_correct else '❌ SOME FAILED'}")

    # Run benchmark if correctness passes
    if all_correct and test_data[0].device.type == 'cuda':
        benchmark_implementations(algorithm, test_data)

        print(f"\n🎉 {algorithm.upper()} Algorithm Test Completed!")
        print(f"\nWhat we verified:")
        print("✅ Numerical correctness against PyTorch reference")
        print("✅ CPU implementation accuracy")
        print(f"✅ GPU {algorithm} implementation accuracy")
        print("✅ Performance comparison CPU vs GPU")

        print(f"\nLearning outcomes:")
        print(f"- {algorithm.title()} kernel implementation and optimization")
        print("- Cross-platform correctness verification")
        print("- Performance characterization and bottleneck analysis")
        if algorithm == "unfused":
            print("- Multi-kernel pipeline composition")
            print("- Memory bandwidth vs compute trade-offs")
        else:
            print("- Single-kernel fusion benefits and limitations")
            print("- Computation density optimization")
    else:
        if not all_correct:
            print(f"\n❌ Correctness issues found - skipping benchmark")
        else:
            print(f"\n⚠️  CUDA not available - only correctness tested")

def main():
    """Main function with command line argument handling."""
    parser = argparse.ArgumentParser(description="Run tests for Puzzle 20")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fused", action="store_true", help="Test and benchmark fused algorithm only")
    group.add_argument("--unfused", action="store_true", help="Test and benchmark unfused algorithm only")
    group.add_argument("--fused-only", action="store_true", help="Run only the fused kernel test (legacy)")
    args = parser.parse_args()

    if args.fused:
        run_algorithm_specific_test("fused")
    elif args.unfused:
        run_algorithm_specific_test("unfused")
    elif args.fused_only:
        run_fused_only_test()
    else:
        run_comprehensive_test()

if __name__ == "__main__":
    main()
