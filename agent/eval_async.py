"""Async batch evaluation of agent trajectories (parallel vLLM calls)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset_utils import load_jsonl
from .eval import EVAL_SYSTEM_PROMPT, _parse_eval_response
from .utils import extract_final_answer, trajectory_stats
from .vllm_client_async import VLLMClientAsync


async def evaluate_trajectories(
    records: List[Dict[str, Any]],
    dataset_path: str,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    eval_batch_size: int = 16,
    temperature: float = 0.0,
    max_tokens: int = 256,
    output_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parallel eval over ``records`` using ``asyncio.gather``."""
    dataset = load_jsonl(dataset_path)
    gold_map = {row["query_id"]: row["answer"] for row in dataset}
    question_map = {row["query_id"]: row.get("query", "") for row in dataset}

    client = VLLMClientAsync(base_url=base_url, api_key=api_key)
    details: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    async def _eval_one(sub: Dict[str, Any]) -> Dict[str, Any]:
        qid = sub.get("query_id", "")
        gold = gold_map.get(qid, "")
        question = question_map.get(qid, "")
        pred = sub.get("predicted_answer", "")

        if not pred:
            pred = extract_final_answer(sub.get("messages", [])) or ""

        if not gold:
            return {
                "query_id": qid, "question": question,
                "gold_answer": "", "predicted_answer": pred,
                "eval_judgment": "INCORRECT",
                "eval_reasoning": "No gold answer found.",
                "eval_model_response": "",
                "trajectory_stats": {},
                "status": sub.get("status", "unknown"),
            }

        if not pred:
            return {
                "query_id": qid, "question": question,
                "gold_answer": gold, "predicted_answer": "",
                "eval_judgment": "INCORRECT",
                "eval_reasoning": "No predicted answer.",
                "eval_model_response": "",
                "trajectory_stats": trajectory_stats(sub.get("messages", [])),
                "status": sub.get("status", "unknown"),
            }

        eval_msgs = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"},
        ]

        try:
            resp = await client.simple_chat(
                model=model, messages=eval_msgs,
                temperature=temperature, max_tokens=max_tokens,
            )
            eval_text = resp["choices"][0]["message"]["content"]
            judgment, reasoning = _parse_eval_response(eval_text)
        except Exception as e:
            eval_text = f"ERROR: {e}"
            judgment, reasoning = "INCORRECT", str(e)

        return {
            "query_id": qid, "question": question,
            "gold_answer": gold, "predicted_answer": pred,
            "eval_judgment": judgment,
            "eval_reasoning": reasoning,
            "eval_model_response": eval_text,
            "trajectory_stats": trajectory_stats(sub.get("messages", [])),
            "status": sub.get("status", "unknown"),
        }

    try:
        for batch_start in range(0, len(records), eval_batch_size):
            batch = records[batch_start:batch_start + eval_batch_size]
            results = await asyncio.gather(*[_eval_one(r) for r in batch])
            for d in results:
                if d["eval_judgment"] == "CORRECT":
                    correct += 1
                total += 1
                details.append(d)

            done = min(batch_start + eval_batch_size, len(records))
            print(f"[eval] {done}/{len(records)} done", flush=True)
    finally:
        await client._client.close()

    accuracy = correct / total if total > 0 else 0.0
    all_tc = [d["trajectory_stats"].get("num_tool_calls", 0) for d in details]
    all_docs = [d["trajectory_stats"].get("num_retrieved_docs", 0) for d in details]

    summary: Dict[str, Any] = {
        "total_queries": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "avg_tool_calls_per_query": round(sum(all_tc) / total, 2) if total > 0 else 0,
        "avg_retrieved_docs_per_query": round(sum(all_docs) / total, 2) if total > 0 else 0,
        "total_tool_calls": sum(all_tc),
        "total_retrieved_docs": sum(all_docs),
        "eval_model": model,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[eval] saved results → {output_path}", flush=True)

    return summary, details
