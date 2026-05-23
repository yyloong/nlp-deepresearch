"""
将 distill/data_train.parquet 中的干净8534条数据
转换成 Qwen3-8B 工具调用格式，保存为 train/sft_data.jsonl

原始消息角色:
  system / user / reasoning / tool_call / tool_output / answer

转换后 Qwen3 格式:
  system(带工具定义) / user / assistant(<think>+<tool_call>) / tool / ... / assistant(<think>+最终答案)
"""

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent

# ──────────────────────────────────────────────
# 工具定义（Qwen3 标准 function-calling 格式）
# ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches a document corpus and returns relevant snippets with docids and scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["user_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieves the full text of a document by its docid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": "The document ID to retrieve",
                    }
                },
                "required": ["docid"],
            },
        },
    },
]

# Qwen3 system prompt with tools embedded (手动构造，与 apply_chat_template(tools=...) 输出一致)
_TOOLS_STR = "\n".join(json.dumps(t, ensure_ascii=False) for t in TOOLS)
SYSTEM_PROMPT = f"""You are a deep research assistant. Answer complex questions through iterative reasoning and information retrieval using the available tools. Think carefully before each action and support every claim with retrieved evidence.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{_TOOLS_STR}
</tools>"""


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def convert_tool_call(content: str) -> str:
    """将 tool_call 中的 'parameters' key 改为 'arguments'（Qwen3 规范）。"""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return content
    try:
        obj = json.loads(m.group(1))
        if "parameters" in obj:
            obj["arguments"] = obj.pop("parameters")
        return f"<tool_call>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_call>"
    except (json.JSONDecodeError, ValueError):
        return content


def strip_tags(content: str, tag: str) -> str:
    """去掉 <tag>...</tag> 包裹，返回内部文本。"""
    return re.sub(rf"</?{tag}>", "", content).strip()


def raw_to_qwen3(raw_messages: list) -> list | None:
    """
    将原始多角色消息列表转换为 Qwen3 工具调用格式。

    转换规则:
      reasoning + tool_call  →  assistant: <think>...</think>\n<tool_call>...</tool_call>
      tool_output            →  tool:  result_text  (去掉 <tool_response> 包裹，训练时手动加回)
      reasoning + answer     →  assistant: <think>...</think>\nfinal_answer_text
    """
    qwen3: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    i = 0
    # 跳过原始 system 消息
    while i < len(raw_messages) and raw_messages[i]["role"] == "system":
        i += 1

    while i < len(raw_messages):
        role = raw_messages[i]["role"]
        content = raw_messages[i]["content"]

        if role == "user":
            qwen3.append({"role": "user", "content": content})
            i += 1

        elif role == "reasoning":
            # reasoning 的 content 已经是 <think>...</think>
            think_block = content
            next_role = raw_messages[i + 1]["role"] if i + 1 < len(raw_messages) else None

            if next_role == "tool_call":
                tool_content = convert_tool_call(raw_messages[i + 1]["content"])
                qwen3.append({
                    "role": "assistant",
                    "content": f"{think_block}\n{tool_content}",
                })
                i += 2

            elif next_role == "answer":
                answer_text = strip_tags(raw_messages[i + 1]["content"], "answer")
                qwen3.append({
                    "role": "assistant",
                    "content": f"{think_block}\n{answer_text}",
                })
                i += 2

            else:
                # 边缘情况：孤立的 reasoning
                qwen3.append({"role": "assistant", "content": think_block})
                i += 1

        elif role == "tool_output":
            # 去掉 <tool_response> 标签，格式化时手动添加（与 Qwen3 template 一致）
            result = strip_tags(content, "tool_response")
            qwen3.append({"role": "tool", "content": result})
            i += 1

        elif role in ("tool_call", "answer"):
            # 已由前置 reasoning 处理，跳过
            i += 1

        else:
            i += 1

    # 验证：至少有 user 和 assistant 各一条
    roles = {m["role"] for m in qwen3}
    if "user" not in roles or "assistant" not in roles:
        return None

    return qwen3


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    # 加载 train query 列表（用于过滤 clean 8534 条）
    all_queries_path = ROOT / "distill" / "browsecomp_plus_all_queries.jsonl"
    hard50_path = ROOT / "browsecomp_plus_hard50.jsonl"

    hard50_ids: set[str] = set()
    with open(hard50_path) as f:
        for line in f:
            hard50_ids.add(json.loads(line)["query_id"])

    train_queries: list[str] = []
    with open(all_queries_path) as f:
        for line in f:
            item = json.loads(line)
            if item["query_id"] not in hard50_ids:
                train_queries.append(normalize(item["query"]))

    print(f"[convert] train queries: {len(train_queries)}")

    # 读取 SFT 数据（已去除 hard50）
    data_path = ROOT / "distill" / "data_train.parquet"
    table = pq.read_table(data_path)
    print(f"[convert] total records in data_train.parquet: {table.num_rows}")

    out_path = Path(__file__).parent / "sft_data.jsonl"
    converted = skipped_no_match = skipped_bad = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for i in range(table.num_rows):
            msg_str = table.column("messages")[i].as_py()
            raw_msgs = json.loads(msg_str)

            # 只保留 user 消息包含 train query 子串的干净记录
            user_content = normalize(
                next((m["content"] for m in raw_msgs if m["role"] == "user"), "")
            )
            if not any(q in user_content for q in train_queries):
                skipped_no_match += 1
                continue

            qwen3_msgs = raw_to_qwen3(raw_msgs)
            if qwen3_msgs is None:
                skipped_bad += 1
                continue

            fout.write(json.dumps({"messages": qwen3_msgs}, ensure_ascii=False) + "\n")
            converted += 1

            if converted % 500 == 0:
                print(f"[convert] {converted} done ...", flush=True)

    print(f"[convert] converted: {converted}")
    print(f"[convert] skipped (no query match): {skipped_no_match}")
    print(f"[convert] skipped (bad format): {skipped_bad}")
    print(f"[convert] saved → {out_path}")


if __name__ == "__main__":
    main()
