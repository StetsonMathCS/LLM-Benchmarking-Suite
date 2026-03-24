"""
test_ollama.py

Test script to run all benchmarks with Ollama provider.
Verifies that the full benchmarking system works end-to-end.

Prerequisites:
  1. Ollama installed and running: ollama serve (or via Docker)
  2. A model pulled: ollama pull llama2 (or your preferred model)
  3. Dependencies: pip install ollama httpx
"""

import sys
from pathlib import Path

# Add codebase to path
sys.path.insert(0, str(Path(__file__).parent))

from core.base import ModelConfig, BenchmarkStatus
from core.suite import TestSuite, SuiteConfig
from core.registry import ProviderRegistry
import time
import json


def test_ollama_connection():
    """Test that Ollama is available and has models."""
    print("\n" + "="*70)
    print("STEP 1: Testing Ollama Connection")
    print("="*70)
    
    try:
        # Create a temporary provider just to test connection
        config = ModelConfig(
            provider="ollama",
            model_name="mistral:7b",
            base_url="http://localhost:1561",
            temperature=0.7,
            max_tokens=512,
        )
        provider = ProviderRegistry.create(config)
        # Test connection
        if not provider.connect():
            print("❌ Failed to import ollama library. Install with: pip install ollama httpx")
            return False
        
        # Test availability
        if not provider.is_available():
            print("❌ Ollama server not available at http://localhost:1561")
            print("   Start Ollama with: ollama serve")
            return False
        
        print("✅ Ollama connection successful")
        
        # List available models
        models = provider.list_local_models()
        if not models:
            print("❌ No models available. Pull a model with: ollama pull llama2")
            return False
        
        print(f"✅ Available models: {', '.join(models)}")        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_single_benchmark():
    """Run a single benchmark to verify the pipeline works."""
    print("\n" + "="*70)
    print("STEP 2: Testing Single Benchmark (Refactoring)")
    print("="*70)
    
    try:
        config = ModelConfig(
            provider="ollama",
            model_name="mistral:7b",
            base_url="http://localhost:1561",
            temperature=0.3,  # Lower for consistency
            max_tokens=1024,
            system_prompt="You are a Python code refactoring expert.",
        )
        
        registry = ProviderRegistry()
        provider = registry.create(config)
        provider.connect()
        
        suite_config = SuiteConfig(
            name="Ollama Refactoring Test",
            language="python",
            selected_benchmarks=["refactoring"],  # Just test refactoring
        )
        
        suite = TestSuite(provider, suite_config)
        suite.register_benchmarks()
        
        print(f"Running benchmarks: {suite_config.selected_benchmarks}")
        print(f"Language: {suite_config.language}")
        print("Starting benchmark suite...")
        
        results = suite.run_all()
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Total results: {len(results)}")
        
        if results:
            for i, result in enumerate(results[:3]):  # Show first 3
                print(f"\n   Result {i+1}:")
                print(f"     - Benchmark: {result.benchmark_name}")
                print(f"     - Status: {result.status}")
                if result.combined_score is not None:
                    print(f"     - Score: {result.combined_score:.2f}")
                if result.issues_found:
                    print(f"     - Issues: {list(result.issues_found.keys())}")
                    for dim_name, error_msg in result.issues_found.items():
                        print(f"       • {dim_name}: {error_msg}")
                if result.llm_response and result.llm_response.error:
                    print(f"     - LLM Error: {result.llm_response.error}")
                if result.details.get("error"):
                    print(f"     - Error Details: {result.details['error']}")
        
        summary = suite.get_summary()
        print(f"\n   Summary:")
        print(f"     - Total: {summary['total']}")
        print(f"     - Passed: {summary['passed']}")
        print(f"     - Failed: {summary['failed']}")
        print(f"     - Pass rate: {summary['pass_rate']:.1%}")
        print(f"     - Time: {summary['elapsed_s']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_benchmarks():
    """Run all benchmark types to verify they all work."""
    print("\n" + "="*70)
    print("STEP 3: Testing All Benchmark Types")
    print("="*70)
    
    try:
        config = ModelConfig(
            provider="ollama",
            model_name="mistral:7b",
            base_url="http://localhost:1561",
            temperature=0.3,
            max_tokens=1024,
        )
        
        registry = ProviderRegistry()
        provider = registry.create(config)
        provider.connect()
        
        suite_config = SuiteConfig(
            name="Ollama Full Test",
            language="python",
            # Run all benchmarks
            selected_benchmarks=[
                "bug_fixing",
                "code_completion",
                "code_generation",
                "code_review",
                "partial_transform",
                "refactoring",
                "test_generation",
                "translation",
            ]
        )
        
        suite = TestSuite(provider, suite_config)
        suite.register_benchmarks()
        
        print(f"Running all {len(suite_config.selected_benchmarks)} benchmark types")
        
        start_time = time.time()
        results = suite.run_all()
        elapsed = time.time() - start_time
        
        summary = suite.get_summary()
        
        print(f"\n✅ Full test suite complete!")
        print(f"\nResults Summary:")
        print(f"  - Total benchmarks run: {summary['total']}")
        print(f"  - Passed: {summary['passed']}")
        print(f"  - Failed: {summary['failed']}")
        print(f"  - Pass rate: {summary['pass_rate']:.1%}")
        print(f"  - Total time: {summary['elapsed_s']:.1f}s ({elapsed:.1f}s measured)")
        
        # Group results by benchmark type
        by_task = {}
        for result in results:
            task = result.benchmark_name
            if task not in by_task:
                by_task[task] = {"passed": 0, "failed": 0, "error": 0}
            
            if result.status == BenchmarkStatus.PASSED:
                by_task[task]["passed"] += 1
            elif result.status == BenchmarkStatus.FAILED:
                by_task[task]["failed"] += 1
            else:
                by_task[task]["error"] += 1
        
        print(f"\nBreakdown by benchmark type:")
        for task, counts in by_task.items():
            total = counts["passed"] + counts["failed"] + counts["error"]
            print(f"  - {task}: {counts['passed']}/{total} passed ({counts['error']} errors)")
        
        return True
        
    except Exception as e:
        print(f"❌ Full benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print(" 🧪 LLM Benchmarking Suite - Ollama Test")
    print("="*70)
    
    # Step 1: Connection test
    if not test_ollama_connection():
        print("\n❌ Cannot proceed without Ollama connection")
        return 1
    
    # Step 2: Single benchmark test
    if not test_single_benchmark():
        print("\n⚠️  Single benchmark test failed, skipping full test")
        return 1
    
    # Step 3: All benchmarks test
    if not test_all_benchmarks():
        print("\n⚠️  Full benchmark test had issues")
        return 1
    
    print("\n" + "="*70)
    print(" ✅ All tests passed!")
    print("="*70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
