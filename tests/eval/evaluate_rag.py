"""RAG evaluation harness for retrieval quality measurement.

Runs queries from the query_set.yaml against the live RAG system and
measures retrieval metrics (results count, score distribution, latency).

Usage:
    cd src && python -m tests.eval.evaluate_rag
    
    Or from project root:
    PYTHONPATH=src python tests/eval/evaluate_rag.py

Metrics computed:
    - results_count: number of results per query
    - avg_score: average relevance score
    - max_score: top result score
    - latency_ms: query execution time
    - has_results: boolean (did we retrieve anything?)

Results are written to tests/eval/results.json for analysis.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_query_set(path: str = None) -> List[Dict[str, Any]]:
    """Load the evaluation query set from YAML.
    
    Args:
        path: Path to query_set.yaml (auto-detected if None)
        
    Returns:
        List of query dicts
    """
    if path is None:
        path = str(Path(__file__).parent / "query_set.yaml")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return data.get("queries", [])


def evaluate_retrieval(rag, queries: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    """Run queries against the RAG system and collect metrics.
    
    Args:
        rag: LlamaIndexRAG instance
        queries: List of query dicts from query_set.yaml
        k: Number of results to retrieve per query
        
    Returns:
        List of result dicts with metrics
    """
    results = []
    
    for q in queries:
        qid = q["id"]
        query = q["query"]
        intent = q.get("intent", "unknown")
        language = q.get("language", "unknown")
        
        # Skip multi-turn queries (they need conversation context)
        if intent == "multi_turn":
            results.append({
                "id": qid,
                "query": query,
                "intent": intent,
                "language": language,
                "skipped": True,
                "reason": "multi_turn requires conversation context",
            })
            continue
        
        # Run the search
        start = time.monotonic()
        try:
            search_results = rag.search(query=query, k=k)
            elapsed_ms = (time.monotonic() - start) * 1000
        except Exception as e:
            results.append({
                "id": qid,
                "query": query,
                "intent": intent,
                "language": language,
                "error": str(e),
                "latency_ms": (time.monotonic() - start) * 1000,
            })
            continue
        
        # Compute metrics
        scores = [
            r.score for r in search_results
            if r.score is not None
        ]
        
        result = {
            "id": qid,
            "query": query,
            "intent": intent,
            "language": language,
            "results_count": len(search_results),
            "has_results": len(search_results) > 0,
            "scores": scores[:5],  # Top 5 scores
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "latency_ms": round(elapsed_ms, 1),
        }
        
        # Include top result metadata for manual inspection
        if search_results:
            top = search_results[0]
            meta = getattr(top.node, "metadata", {}) if top.node else {}
            result["top_result"] = {
                "source": meta.get("source", ""),
                "sender": meta.get("sender", ""),
                "chat_name": meta.get("chat_name", ""),
                "text_preview": (getattr(top.node, "text", "") or "")[:200],
            }
        
        results.append(result)
        print(f"  {qid}: {len(search_results)} results, "
              f"max_score={result['max_score']:.3f}, "
              f"latency={result['latency_ms']:.0f}ms")
    
    return results


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation results.
    
    Args:
        results: List of per-query result dicts
        
    Returns:
        Summary dict with aggregate metrics
    """
    evaluated = [r for r in results if not r.get("skipped") and not r.get("error")]
    
    if not evaluated:
        return {"error": "No queries were evaluated"}
    
    has_results_count = sum(1 for r in evaluated if r.get("has_results"))
    avg_latency = sum(r.get("latency_ms", 0) for r in evaluated) / len(evaluated)
    avg_score = sum(r.get("avg_score", 0) for r in evaluated) / len(evaluated)
    
    # Per-language breakdown
    by_language: Dict[str, List] = {}
    for r in evaluated:
        lang = r.get("language", "unknown")
        by_language.setdefault(lang, []).append(r)
    
    language_stats = {}
    for lang, lang_results in by_language.items():
        has_res = sum(1 for r in lang_results if r.get("has_results"))
        language_stats[lang] = {
            "total": len(lang_results),
            "has_results": has_res,
            "recall_rate": has_res / len(lang_results) if lang_results else 0,
            "avg_score": sum(r.get("avg_score", 0) for r in lang_results) / len(lang_results),
        }
    
    return {
        "total_queries": len(results),
        "evaluated": len(evaluated),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "errors": sum(1 for r in results if r.get("error")),
        "has_results_rate": has_results_count / len(evaluated),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_score": round(avg_score, 4),
        "by_language": language_stats,
    }


def main():
    """Run the evaluation harness."""
    # Ensure src is on the path
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print("Loading RAG system...")
    from llamaindex_rag import get_rag
    rag = get_rag()
    
    print("Loading query set...")
    queries = load_query_set()
    print(f"Loaded {len(queries)} queries")
    
    print("\nRunning evaluation...")
    results = evaluate_retrieval(rag, queries)
    
    summary = compute_summary(results)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries:     {summary['total_queries']}")
    print(f"Evaluated:         {summary['evaluated']}")
    print(f"Has results rate:  {summary.get('has_results_rate', 0):.1%}")
    print(f"Avg latency:       {summary.get('avg_latency_ms', 0):.0f}ms")
    print(f"Avg score:         {summary.get('avg_score', 0):.4f}")
    
    if "by_language" in summary:
        print("\nPer-language breakdown:")
        for lang, stats in summary["by_language"].items():
            print(f"  {lang}: recall={stats['recall_rate']:.1%}, "
                  f"avg_score={stats['avg_score']:.4f}, "
                  f"n={stats['total']}")
    
    # Save results
    output_path = str(Path(__file__).parent / "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
