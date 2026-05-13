#!/usr/bin/env python3
"""
Pretty-print a submission JSONL file for human inspection.

Usage:
    python scripts/pprint_submission.py runs/submission_20260513_124203.jsonl
    python scripts/pprint_submission.py runs/submission_20260513_124203.jsonl --limit 5
    python scripts/pprint_submission.py runs/submission_20260513_124203.jsonl -e runs/eval_20260513_124203.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SEP = "─" * 80
THIN = "·" * 80


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _trunc(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _tool_result_summary(content: str) -> str:
    """Summarise a tool result: count docs, show first snippet."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _trunc(content, 120)
    if isinstance(data, list):
        return f"  [{len(data)} docs]  eg. {_trunc(json.dumps(data[0], ensure_ascii=False), 120)}"
    if isinstance(data, dict):
        if "error" in data:
            return f"  ❌ ERROR: {_trunc(data['error'], 120)}"
        if "docid" in data:
            return f"  [1 doc]  docid={data.get('docid','?')}  {_trunc(data.get('text','')[:80], 80)}"
        return f"  [dict]  {_trunc(json.dumps(data, ensure_ascii=False), 120)}"
    return _trunc(str(data), 120)


def pprint_submission(
    records: List[Dict[str, Any]],
    eval_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    eval_by_qid: Dict[str, Dict[str, Any]] = {}
    if eval_records:
        for r in eval_records:
            qid = r.get("query_id", "")
            if qid:
                eval_by_qid[qid] = r

    for idx, rec in enumerate(records):
        qid = rec.get("query_id", "?")
        answer = rec.get("predicted_answer", "")
        msgs: List[Dict[str, Any]] = rec.get("messages", [])

        # ── question ──
        question = ""
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content", "")
                if content and not content.startswith("[PROGRESS SUMMARY"):
                    question = content
                    break

        # ── stats ──
        tool_calls_cnt = 0
        searches: List[str] = []
        get_doc_ids: List[str] = []
        for m in msgs:
            if m.get("role") == "tool":
                tool_calls_cnt += 1
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    if fn.get("name") == "search":
                        try:
                            a = json.loads(fn.get("arguments", "{}"))
                            searches.append(a.get("query", "?"))
                        except json.JSONDecodeError:
                            searches.append(fn.get("arguments", "?")[:60])
                    elif fn.get("name") == "get_document":
                        try:
                            a = json.loads(fn.get("arguments", "{}"))
                            get_doc_ids.append(a.get("docid", "?"))
                        except json.JSONDecodeError:
                            get_doc_ids.append("?")

        # ── eval info ──
        ev = eval_by_qid.get(qid, {})

        # ── print ──
        print(f"\n{SEP}")
        print(f"  [{idx + 1}/{len(records)}]  query_id = {qid}")
        print(f"{SEP}")

        print(f"  📋 Question  : {_trunc(question, 180)}")
        if ev:
            gold = ev.get("gold_answer", "")
            judgment = ev.get("eval_judgment", "?")
            mark = "✅" if judgment == "CORRECT" else "❌"
            print(f"  🎯 Gold      : {gold}")
            print(f"  🤖 Predicted : {_trunc(answer, 180)}")
            print(f"  {mark} Judgment  : {judgment}")
        else:
            print(f"  🤖 Predicted : {_trunc(answer, 180)}")

        print(f"  🔧 Tool calls: {tool_calls_cnt} (search={len(searches)}, get_doc={len(get_doc_ids)})")

        if searches:
            print(f"  {THIN}")
            print("  Searches:")
            for s in searches:
                print(f"    → {_trunc(s, 120)}")
        if get_doc_ids:
            print(f"  {THIN}")
            print(f"  Documents retrieved: {', '.join(get_doc_ids[:8])}")

        # ── tool results (summarised) ──
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        if tool_msgs:
            print(f"  {THIN}")
            print(f"  Tool results ({len(tool_msgs)} messages):")
            for tm in tool_msgs:
                cid = tm.get("tool_call_id", "?")[-12:]
                summary = _tool_result_summary(tm.get("content", ""))
                print(f"    [{cid}] {summary}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pretty-print a deep-research submission")
    p.add_argument("submission", help="Path to submission JSONL")
    p.add_argument("--limit", "-n", type=int, default=None, help="Max entries to print")
    p.add_argument("--eval", "-e", default=None, help="Path to eval JSONL (for gold answers & judgments)")
    args = p.parse_args()

    records = load_jsonl(args.submission)
    if args.limit:
        records = records[: args.limit]

    eval_records = None
    if args.eval:
        eval_records = load_jsonl(args.eval)

    pprint_submission(records, eval_records)


if __name__ == "__main__":
    main()
