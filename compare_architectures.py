# -*- coding: utf-8 -*-
"""
compare_architectures.py - Compare Single-Agent vs Multi-Agent Performance
Purpose: Demonstrate the benefits of multi-agent CrewAI architecture
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_pipeline(script_name, description):
    """Run a pipeline and measure execution time"""
    print(f"\n🚀 Running: {description}")
    print(f"   Script: {script_name}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.2f}s")
        return {
            "success": True,
            "time": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.2f}s")
        return {
            "success": False,
            "time": elapsed,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "time": elapsed,
            "error": str(e)
        }

def load_json_safe(filepath):
    """Safely load JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {"error": f"File not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to load {filepath}: {e}"}

def analyze_outputs():
    """Compare outputs from both architectures"""
    print_header("📊 ANALYZING OUTPUTS")
    
    # Load single-agent output
    single_output = load_json_safe("single_agent_output.json")
    
    # Load multi-agent outputs
    rag_output = load_json_safe("rag_output.json")
    fusion_output = load_json_safe("fusion_output_agent.json")
    explainer_output = load_json_safe("explainer_output.json")
    consensus_output = load_json_safe("consensus_output.json")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "architectures": {
            "single_agent": {
                "files_generated": 1,
                "execution_time": single_output.get("total_execution_time", "N/A"),
                "stages_completed": single_output.get("completed_stages", "N/A"),
                "total_stages": single_output.get("total_stages", "N/A"),
                "vulnerability_detected": single_output.get("final_output", {}).get("vulnerability_type", "N/A"),
                "has_error": "error" in single_output
            },
            "multi_agent": {
                "files_generated": 4,
                "agents_used": ["RAG Agent", "Embedding Agent", "Fusion Agent", "Explainer Agent", "Consensus Agent"],
                "vulnerability_detected": fusion_output.get("Predict", "N/A"),
                "rag_success": "error" not in rag_output,
                "fusion_success": "error" not in fusion_output,
                "explainer_success": "error" not in explainer_output,
                "consensus_success": "error" not in consensus_output
            }
        },
        "key_differences": {
            "modularity": {
                "single_agent": "All logic in one monolithic tool - hard to maintain",
                "multi_agent": "Specialized agents with clear responsibilities - easy to modify individual components"
            },
            "error_handling": {
                "single_agent": "Single point of failure - one error stops everything",
                "multi_agent": "Isolated failures - if one agent fails, others can still provide partial results"
            },
            "model_specialization": {
                "single_agent": "Uses same LLM for all tasks - suboptimal",
                "multi_agent": "Different models for different tasks (Gemini, Unsloth, Ollama, Transformer) - optimal"
            },
            "maintainability": {
                "single_agent": "Must understand entire codebase to make changes",
                "multi_agent": "Can update individual agents independently"
            },
            "scalability": {
                "single_agent": "Adding new features requires modifying monolithic tool",
                "multi_agent": "Add new agents without touching existing ones"
            },
            "debugging": {
                "single_agent": "Hard to isolate issues - everything is intertwined",
                "multi_agent": "Easy to debug - test each agent independently"
            }
        }
    }
    
    # Save comparison
    with open("architecture_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("\n📈 COMPARISON RESULTS:")
    print(f"\n1. FILES GENERATED:")
    print(f"   Single-Agent: 1 output file")
    print(f"   Multi-Agent:  4 output files (specialized outputs)")
    
    print(f"\n2. VULNERABILITY DETECTION:")
    print(f"   Single-Agent: {comparison['architectures']['single_agent']['vulnerability_detected']}")
    print(f"   Multi-Agent:  {comparison['architectures']['multi_agent']['vulnerability_detected']}")
    
    print(f"\n3. EXECUTION TIME:")
    print(f"   Single-Agent: {comparison['architectures']['single_agent']['execution_time']}s")
    print(f"   Multi-Agent:  Check main_pipeline.py output for total time")
    
    print(f"\n4. KEY ADVANTAGES OF MULTI-AGENT:")
    for key, diff in comparison['key_differences'].items():
        print(f"\n   {key.upper()}:")
        print(f"   ❌ Single: {diff['single_agent']}")
        print(f"   ✅ Multi:  {diff['multi_agent']}")
    
    print(f"\n✅ Detailed comparison saved to: architecture_comparison.json")
    
    return comparison

def main():
    print_header("🔬 ARCHITECTURE COMPARISON STUDY")
    print("Comparing Single-Agent Monolithic vs Multi-Agent CrewAI")
    print("Sample: contracts/sample.sol")
    
    results = {}
    
    # Test 1: Single-Agent Baseline
    print_header("TEST 1: SINGLE-AGENT MONOLITHIC APPROACH")
    results['single_agent'] = run_pipeline(
        "single_agent_baseline.py",
        "Single Agent Baseline (Everything in one agent)"
    )
    
    # Test 2: Multi-Agent CrewAI
    print_header("TEST 2: MULTI-AGENT CREWAI APPROACH")
    results['multi_agent'] = run_pipeline(
        "main_pipeline.py",
        "Multi-Agent CrewAI Pipeline (Specialized agents)"
    )
    
    # Analyze and compare
    if results['single_agent']['success'] or results['multi_agent']['success']:
        comparison = analyze_outputs()
    else:
        print("\n⚠️ Both pipelines failed - cannot compare outputs")
    
    # Generate summary
    print_header("📋 EXECUTION SUMMARY")
    
    print(f"\nSingle-Agent Status: {'✅ Success' if results['single_agent']['success'] else '❌ Failed'}")
    print(f"Single-Agent Time:   {results['single_agent']['time']:.2f}s")
    
    print(f"\nMulti-Agent Status:  {'✅ Success' if results['multi_agent']['success'] else '❌ Failed'}")
    print(f"Multi-Agent Time:    {results['multi_agent']['time']:.2f}s")
    
    print_header("🎯 CONCLUSION FOR THESIS")
    print("""
The multi-agent architecture using CrewAI provides several critical advantages:

1. **SPECIALIZATION**: Each agent uses optimal models for its task
   - RAG Agent: Gemini + Unsloth + ChromaDB
   - Fusion Agent: Custom Transformer model
   - Explainer: Ollama for natural language
   - Consensus: Validation with KB retrieval

2. **MODULARITY**: Easy to modify/upgrade individual components
   - Can swap Gemini → GPT-4 without touching other agents
   - Can upgrade fusion model independently

3. **FAULT ISOLATION**: Partial results available even if one agent fails
   - RAG can succeed even if Fusion fails
   - Consensus can work with partial inputs

4. **MAINTAINABILITY**: Clear separation of concerns
   - Each agent has single responsibility
   - Easy to test agents independently

5. **SCALABILITY**: Simple to add new analysis types
   - Add new agents without modifying existing ones
   - No risk of breaking existing functionality

The single-agent approach forces all logic into one monolithic tool,
making it harder to maintain, debug, and scale.
    """)
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)

if __name__ == "__main__":
    main()
