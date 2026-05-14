"""
自动评估脚本：使用 LLM 判断 agent 预测答案与标准答案是否一致。

用法：
    # 命令行
    python -m agent.eval \
        --submission runs/submission.jsonl \
        --dataset browsecomp_plus_hard50.jsonl \
        --model Qwen3-8B \
        --base-url http://127.0.0.1:8000/v1 \
        --output runs/eval_results.jsonl

    # notebook 中调用
    from agent.eval import run_evaluation
    summary, details = run_evaluation(
        submission_path="runs/submission.jsonl",
        dataset_path="browsecomp_plus_hard50.jsonl",
        model_name="Qwen3-8B",
        base_url="http://127.0.0.1:8000/v1",
        output_path="runs/eval_results.jsonl",
    )
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset_utils import load_jsonl
from .vllm_client import VLLMClient


# ── Eval prompt ──────────────────────────────────────────────
EVAL_SYSTEM_PROMPT = """You are an expert evaluator for a document-retrieval question-answering system.
The system searches a document corpus to answer complex questions. The "gold answer" IS the correct
answer that CAN be found through searching the corpus. Your task is to judge whether a predicted
answer is semantically equivalent to the gold answer.

CRITICAL RULES:
- The gold answer is the ground truth — if the predicted answer differs from it, the prediction is WRONG.
- If the predicted answer says anything like "cannot be determined", "insufficient evidence",
  "not explicitly named", "not present in the documents", "information is insufficient",
  "cannot be answered", or any similar statement that declines to answer, it is ALWAYS INCORRECT
  (because the gold answer proves that a specific answer does exist in the corpus).
- Do NOT reason about whether the question itself contains enough information — the task
  is to retrieve the answer from documents, not from the question text.

Standard matching rules:
- Ignore case differences, punctuation variations, and extra whitespace.
- Treat abbreviations and full forms as equivalent (e.g., "US" = "United States").
- If the predicted answer contains the gold answer as a substring (or vice versa) and the extra content does not change the meaning, treat as CORRECT.
- If the predicted answer is a valid alternative phrasing of the gold answer, treat as CORRECT.
- If the predicted answer is clearly wrong, incomplete in a meaningful way, or contradicts the gold answer, treat as INCORRECT.

Reply in exactly this format:
Judgment: CORRECT
Reasoning: <one sentence explaining your decision>"""


def _build_eval_user_message(gold_answer: str, predicted_answer: str, question: str = "") -> str:
    parts = []
    if question:
        parts.append(f"Question: {question}")
    parts.append(f"Gold answer: {gold_answer}")
    parts.append(f"Predicted answer: {predicted_answer}")
    return "\n".join(parts)


def _parse_eval_response(response_text: str) -> Tuple[str, str]:
    """Parse the eval model's response, returning (judgment, reasoning)."""
    judgment = "INCORRECT"
    reasoning = ""

    # Match "Judgment: CORRECT" or "Judgment: INCORRECT"
    jud_match = re.search(r'Judgment:\s*(CORRECT|INCORRECT)', response_text, re.IGNORECASE)
    if jud_match:
        judgment = jud_match.group(1).upper()

    # Match reasoning
    reason_match = re.search(r'Reasoning:\s*(.+?)$', response_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return judgment, reasoning


def _extract_predicted_answer(messages: List[Dict[str, Any]]) -> str:
    """Extract the predicted answer from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"].strip()
    return ""


def _count_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Count total tool_calls across all assistant messages."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count


def _extract_retrieved_docids(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract all docids from tool responses."""
    docids = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "docid" in item:
                            docids.append(item["docid"])
                elif isinstance(parsed, dict) and "docid" in parsed:
                    docids.append(parsed["docid"])
    return docids


def _compute_trajectory_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from a single trajectory."""
    tool_call_count = _count_tool_calls(messages)
    retrieved_docids = _extract_retrieved_docids(messages)
    num_assistant_msgs = sum(1 for m in messages if m.get("role") == "assistant")
    num_tool_msgs = sum(1 for m in messages if m.get("role") == "tool")

    return {
        "num_tool_calls": tool_call_count,
        "num_assistant_messages": num_assistant_msgs,
        "num_tool_messages": num_tool_msgs,
        "num_retrieved_docs": len(retrieved_docids),
        "unique_retrieved_docids": len(set(retrieved_docids)),
        "retrieved_docids": list(set(retrieved_docids)),
    }


def run_evaluation(
    submission_path: str,
    dataset_path: str,
    model_name: str = "Qwen3-8B",
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    output_path: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    运行自动评估。

    Parameters
    ----------
    submission_path : str
        学生提交的 trajectory 文件路径 (submission.jsonl)。
    dataset_path : str
        原始数据集路径（包含 gold answer）。
    model_name : str
        用于评估的模型名称。
    base_url : str
        vLLM 服务地址。
    api_key : str
        API key。
    output_path : str, optional
        评估结果输出路径。
    temperature : float
        评估模型 temperature。
    max_tokens : int
        评估模型 max_tokens。
    verbose : bool
        是否打印进度。

    Returns
    -------
    summary : dict
        包含 accuracy、总体统计等。
    details : list[dict]
        每个 query 的详细评估结果。
    """
    # 加载数据
    submissions = load_jsonl(submission_path)
    dataset = load_jsonl(dataset_path)

    # 建立 query_id -> gold answer 的映射
    gold_map: Dict[str, str] = {}
    gold_question_map: Dict[str, str] = {}
    for row in dataset:
        gold_map[row["query_id"]] = row["answer"]
        gold_question_map[row["query_id"]] = row.get("query", "")

    client = VLLMClient(base_url=base_url, api_key=api_key)
    details: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0

    for sub in submissions:
        query_id = sub.get("query_id", "")
        gold_answer = gold_map.get(query_id, "")
        question = gold_question_map.get(query_id, "")

        if not gold_answer:
            if verbose:
                print(f"[WARN] query_id={query_id}: no gold answer found in dataset, skipping")
            continue

        messages = sub.get("messages", [])
        predicted_answer = sub.get("predicted_answer", "")
        if not predicted_answer:
            predicted_answer = _extract_predicted_answer(messages)

        eval_text = ""
        if not predicted_answer:
            judgment = "INCORRECT"
            reasoning = "No predicted answer found in submission."
        else:
            # 调用 eval 模型
            eval_messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": _build_eval_user_message(gold_answer, predicted_answer, question)},
            ]
            try:
                response = client.simple_chat(
                    model=model_name,
                    messages=eval_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                eval_text = response["choices"][0]["message"]["content"]
                judgment, reasoning = _parse_eval_response(eval_text)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] query_id={query_id}: eval model call failed: {e}")
                eval_text = f"ERROR: {e}"
                judgment = "INCORRECT"
                reasoning = str(e)

        if judgment == "CORRECT":
            correct_count += 1
        total_count += 1

        # 轨迹统计
        traj_stats = _compute_trajectory_stats(messages)

        detail = {
            "query_id": query_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "eval_judgment": judgment,
            "eval_reasoning": reasoning,
            "eval_model_response": eval_text,
            "trajectory_stats": traj_stats,
            "status": sub.get("status", "unknown"),
        }
        details.append(detail)

        if verbose:
            print(f"[{query_id}] {judgment:>9s} | pred={predicted_answer[:60]}...")

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # 汇总统计
    all_tool_calls = [d["trajectory_stats"]["num_tool_calls"] for d in details]
    all_retrieved = [d["trajectory_stats"]["num_retrieved_docs"] for d in details]

    summary: Dict[str, Any] = {
        "total_queries": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": round(accuracy, 4),
        "avg_tool_calls_per_query": round(sum(all_tool_calls) / total_count, 2) if total_count > 0 else 0,
        "avg_retrieved_docs_per_query": round(sum(all_retrieved) / total_count, 2) if total_count > 0 else 0,
        "total_tool_calls": sum(all_tool_calls),
        "total_retrieved_docs": sum(all_retrieved),
        "eval_model": model_name,
    }

    # 输出结果文件
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            # 第一行写 summary
            f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
            # 后续行写每个 query 的详情
            for detail in details:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\n{'='*50}")
        print(f"Evaluation complete!")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
        print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
        if output_path:
            print(f"Results saved to: {output_path}")

    return summary, details


def main():
    parser = argparse.ArgumentParser(description="自动评估 agent 预测结果")
    parser.add_argument("--submission", required=True, help="学生提交的 trajectory 文件 (submission.jsonl)")
    parser.add_argument("--dataset", required=True, help="原始数据集 (browsecomp_plus_hard50.jsonl)")
    parser.add_argument("--model", default="Qwen3-8B", help="用于评估的模型名称")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 服务地址")
    parser.add_argument("--api-key", default="dummy", help="API key")
    parser.add_argument("--output", default=None, help="评估结果输出路径")
    parser.add_argument("--temperature", type=float, default=0.0, help="评估模型 temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="评估模型 max_tokens")
    args = parser.parse_args()

    if args.output is None:
        submission_stem = Path(args.submission).stem
        args.output = str(Path(args.submission).parent / f"{submission_stem}_eval.jsonl")

    run_evaluation(
        submission_path=args.submission,
        dataset_path=args.dataset,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
