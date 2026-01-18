from typing import List, Dict, Any, Tuple
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import load_rag_chain
import time
import json
from pathlib import Path



EVAL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "What are von Neumann algebras?",
        "expected_topics": ["algebra", "closed", "operator", "hilbert space", "weak topology", "strong topology"],
        "category": "definition"
    },
    {
        "question": "What is the modular operator", 
        "expected_topics": ["modular", "operator", "tomita", "theory", "hilbert space", "state", "adjoint", "von Neumann"],
        "category": "definition"
    },
    { 
        "question": "Describe the process of quantization in physics",
        "expected_topics": ["classical", "quantum", "observables", "commutation", "operators", "phase space"],
        "category": "explanation"
    },
    {
        "question" : "What ared Dirac fields?",
        "expected_topics": ["spinor", "fermion", "relativistic", "equation", "quantum field theory"],
        "category": "definition"
    },
    {
        "question": "Explain the condition needed for a manifold to be spin.",
        "expected_topics": ["spin structure", "manifold", "tangent bundle", "clifford algebra", "second stiefel-whitney class"],
        "category": "explanation"
    },
    {
        "question": "What is a smooth manifold?",
        "expected_topics": ["topological manifold", "atlas", "charts", "differentiable", "smooth structure"],
        "category": "definition"
    }, 
    {
        "question": "Describe the modular automorphism group in von Neumann algebras.",
        "expected_topics": ["modular", "automorphism", "group", "von Neumann", "algebra", "state", "tomita-takesaki"],
        "category": "explanation"
    },
    {
        "question" : "Explain cohomology in algebraic topology.",
        "expected_topics": ["cohomology", "algebraic topology", "chain complex", "cocycles", "coboundaries", "homology"],
        "category": "explanation"   
    },
    {
        "question" : "Motivate and explain the quantization of a the scalar field.",
        "expected_topics": ["quantization", "scalar field", "classical", "quantum", "hamiltonian", "CCR", "Fock space"],
        "category": "explanation"
    }
    # Add more based on your corpus
]


def evaluate_retrieval_quality(
    chain,
    questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    
    results: List[Dict[str, Any]] = []
    
    for i, test_case in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(questions)}] {test_case['question']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = chain.invoke({"input": test_case['question']})
        latency = time.time() - start_time
        
        answer: str = result['answer'].lower()
        sources = result['context']
        
        # Check if expected topics appear in answer
        topics_found = [
            topic for topic in test_case['expected_topics']
            if topic.lower() in answer
        ]
        topic_coverage = len(topics_found) / len(test_case['expected_topics'])
        
        # Check source diversity
        unique_sources = len(set(doc.metadata.get('source') for doc in sources))
        
        # Answer length (should be substantive)
        answer_length = len(answer.split())
        
        eval_result = {
            "question": test_case['question'],
            "category": test_case['category'],
            "answer_length_words": answer_length,
            "topic_coverage": topic_coverage,
            
            "topics_found": topics_found,
            "topics_missing": [
                t for t in test_case['expected_topics'] 
                if t.lower() not in answer
            ],
            "num_sources": len(sources),
            "unique_sources": unique_sources,
            "latency_seconds": latency,
            "answer_preview": result['answer'][:200] + "..."
        }
        
        results.append(eval_result)
        
        # Print summary
        print(f"✓ Topic coverage: {topic_coverage:.1%}")
        print(f"✓ Sources used: {unique_sources} unique / {len(sources)} total")
        print(f"✓ Latency: {latency:.2f}s")
        print(f"✓ Answer length: {answer_length} words")
    
    # Aggregate metrics
    avg_topic_coverage = sum(r['topic_coverage'] for r in results) / len(results)
    avg_latency = sum(r['latency_seconds'] for r in results) / len(results)
    avg_answer_length = sum(r['answer_length_words'] for r in results) / len(results)
    avg_sources = sum(r['num_sources'] for r in results) / len(results)
    
    summary = {
        "total_questions": len(results),
        "avg_topic_coverage": avg_topic_coverage,
        "avg_latency_seconds": avg_latency,
        "avg_answer_length_words": avg_answer_length,
        "avg_sources_retrieved": avg_sources,
        "detailed_results": results
    }
    
    return summary


def save_evaluation_results(results: Dict[str, Any], output_filename: str = "eval_results.json") -> None:
    """Safely resolve and save evaluation results"""
  
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir.parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print formatted summary of evaluation results"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions evaluated: {results['total_questions']}")
    print(f"Average topic coverage: {results['avg_topic_coverage']:.1%}")
    print(f"Average latency: {results['avg_latency_seconds']:.2f}s")
    print(f"Average answer length: {results['avg_answer_length_words']:.0f} words")
    print(f"Average sources retrieved: {results['avg_sources_retrieved']:.1f}")
    print(f"{'='*80}\n")
    
    # Show which questions had poor topic coverage
    poor_coverage = [
        r for r in results['detailed_results']
        if r['topic_coverage'] < 0.5
    ]
    
    if poor_coverage:
        print("Questions with low topic coverage (<50%):")
        for r in poor_coverage:
            print(f"  - {r['question']}")
            print(f"    Missing topics: {', '.join(r['topics_missing'])}")
    else:
        print("✓ All questions achieved >50% topic coverage")


if __name__ == "__main__":
    print("Loading RAG chain...")
    chain = load_rag_chain()
    
    print(f"\nRunning evaluation with {len(EVAL_QUESTIONS)} test questions...")
    results = evaluate_retrieval_quality(chain, EVAL_QUESTIONS)
    
    print_summary(results)
    save_evaluation_results(results)
    
    print("\nTo customize evaluation:")
    print("1. Edit EVAL_QUESTIONS in src/evaluate.py")
    print("2. Add questions specific to your paper content")
    print("3. Define expected topics for each question")