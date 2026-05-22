#!/usr/bin/env python3
"""Test judge on Q549 samples."""
import asyncio, json, os, sys, yaml, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, local_files_only=True)

from agent.agent import Agent
from agent.tool_func import ToolRegistry
from agent.vllm_client_async import VLLMClientAsync
from agent.browsecomp_searcher import BrowseCompBM25Searcher
from agent.utils import truncate_utf8_prefix_to_token_budget
import sqlite3

Q549 = ""
with open("browsecomp_plus_hard50.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["query_id"] == "549":
            Q549 = r["query"]
            break

conn = sqlite3.connect("file:indexes/browsecomp_plus_bm25.sqlite?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

async def test_one(registry, client, name, query, docid, ideal):
    def _factory(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model"] = "qwen_auto"
        cfg["max_tokens"] = 4096
        p = f"/tmp/jt_{os.path.basename(config_path)}"
        with open(p, "w") as f: yaml.dump(cfg, f)
        return Agent(p, client=client, tokenizer=_tok)

    registry._agent_factory = _factory
    judge = _factory("configs/relevance_judge_agent.yaml")
    judge.tool_registry = registry.build_registry("relevance_judge")
    judge.name = f"judge_{docid}"

    row = conn.execute("SELECT text FROM documents WHERE docid=?", (docid,)).fetchone()
    text = row["text"]

    best_pos = 0
    for term in query.lower().split():
        pos = text.lower().find(term)
        if pos >= 0:
            best_pos = pos
            break
    ctx_text = text[max(0, best_pos - 500):]
    ctx_text = truncate_utf8_prefix_to_token_budget(_tok, ctx_text, 6000)

    prompt = (
        f"QUESTION:\n{Q549}\n\n"
        f"(Retrieved by query: \"{query}\")\n\n"
        f"DOCUMENT (docid={docid}):\n{ctx_text}\n\n"
        f"Cross-check: search the entity + other clues from the question. "
        f"Then judge_relevance: HELPFUL, CONFUSING, or IRRELEVANT."
    )

    traj = await judge.run(prompt)

    verdict = "?"
    summary = ""
    tools_used = []
    for m in traj:
        if m["role"] == "assistant":
            for t in (m.get("tool_calls") or []):
                fn = t["function"]["name"]
                args = t["function"].get("arguments", "")[:200]
                if fn == "judge_relevance":
                    try:
                        a = json.loads(t["function"]["arguments"])
                        verdict = a.get("relevance", "?")
                        summary = a.get("summary", "")[:200]
                    except:
                        pass
                else:
                    tools_used.append(f"{fn}({args})")

    ok = "OK" if verdict == ideal else f"FAIL (expected {ideal})"
    print(f"{ok:30s} | {name:40s} | verdict={verdict}")
    if summary:
        print(f"{'':30s} | summary: {summary}")
    if tools_used:
        print(f"{'':30s} | tools: {' | '.join(tools_used)}")
    print()
    return verdict == ideal


async def main():
    client = VLLMClientAsync(base_url="http://127.0.0.1:8000/v1", api_key="dummy", max_concurrent=10)
    searcher = BrowseCompBM25Searcher(index_path="indexes/browsecomp_plus_bm25.sqlite")
    registry = ToolRegistry(searcher=searcher)
    registry._searcher = searcher
    registry.configure_search(search_k=3, snippet_max_chars=1200, use_subagent_summary=False)

    tests = [
        ("Alma Lutz doc (should be HELPFUL)", "librarian partner writer Dakota", "74667", "HELPFUL"),
        ("Herbert Hoover (should be CONFUSING)", "Dakota writer historian", "87163", "CONFUSING"),
        ("Linda Slaughter (should be CONFUSING)", "Dakota writer historian", "45764", "CONFUSING"),
        ("Fargo librarian (should be CONFUSING)", "librarian partner writer Dakota", "13913", "CONFUSING"),
        ("Shipwreck (should be IRRELEVANT)", "librarian partner writer Dakota", "64974", "IRRELEVANT"),
    ]

    correct = 0
    for name, query, docid, ideal in tests:
        if await test_one(registry, client, name, query, docid, ideal):
            correct += 1

    print(f"Accuracy: {correct}/{len(tests)}")
    await client._client.close()


if __name__ == "__main__":
    asyncio.run(main())
