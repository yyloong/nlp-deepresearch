"""
convert_distill.py — 将 run_distill.sh 收集的 DeepSeek 蒸馏轨迹转换为 Qwen3-8B SFT 格式

输入: runs_distill/run_*/trajectories/*.json  (main agent 轨迹)
输出: train/sft_data_distill.jsonl

轨迹格式 (run_serial.py 保存):
  {
    "name": "query_id",
    "agent_type": "main",
    "messages": [<OpenAI 格式 messages>],
    "predicted_answer": "...",
    "status": "submit_answer_confirmed"
  }

SFT 格式 (Qwen3 chat template):
  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "...", "tool_calls": [...]},
      {"role": "tool",      "content": "..."},
      ...
    ]
  }

过滤规则:
  - 只保留 status == "submit_answer_confirmed" 的轨迹
  - 只保留 main agent 轨迹（agent_type == "main"）
  - 去掉 thinking block（<think>...</think>）以节省训练长度（可选）
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def strip_thinking(content: str) -> str:
    """移除 <think>...</think> block（DeepSeek-R1 或 Qwen3 thinking 输出）"""
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()


def convert_message(m: Dict[str, Any], keep_thinking: bool = True) -> Optional[Dict[str, Any]]:
    """
    将 run_serial 格式的消息转为 Qwen3 SFT 消息。
    返回 None 表示跳过该消息。
    """
    role = m.get("role", "")
    content = m.get("content") or ""
    tool_calls = m.get("tool_calls")

    if not keep_thinking and role == "assistant":
        content = strip_thinking(content)

    if role == "system":
        return {"role": "system", "content": content}

    if role == "user":
        return {"role": "user", "content": content}

    if role == "assistant":
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            # tool_calls 已是 OpenAI 格式，直接保留
            msg["tool_calls"] = tool_calls
        return msg

    if role == "tool":
        # tool_call_id 保留用于 Qwen3 chat template
        tool_call_id = m.get("tool_call_id", "")
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }

    return None  # 跳过未知 role


def is_valid_trajectory(messages: List[Dict]) -> bool:
    """检查轨迹结构是否合法（system + user 开头，末尾有 assistant tool_call）"""
    if len(messages) < 3:
        return False
    if messages[0]["role"] != "system":
        return False
    if messages[1]["role"] not in ("user",):
        return False
    # 末尾 assistant 消息应有 tool_calls（submit_answer）
    last_asst = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if last_asst is None:
        return False
    return True


def convert_trajectory(traj: Dict[str, Any], keep_thinking: bool = True) -> Optional[Dict[str, Any]]:
    """将一条 trajectory 转换为 SFT 样本。返回 None 表示跳过。"""
    if traj.get("agent_type") != "main":
        return None
    if traj.get("status") != "submit_answer_confirmed":
        return None

    raw_messages = traj.get("messages", [])
    converted = []
    for m in raw_messages:
        c = convert_message(m, keep_thinking=keep_thinking)
        if c is not None:
            converted.append(c)

    if not is_valid_trajectory(converted):
        return None

    meta = {k: traj[k] for k in traj if k not in ("messages",)}
    return {"messages": converted, **{k: meta[k] for k in ("name",) if k in meta}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepSeek distillation trajectories to Qwen3 SFT format")
    parser.add_argument("--runs-dir",      default="runs_distill",         help="蒸馏 run 目录（含多个 run_*/)")
    parser.add_argument("--output",        default="train/sft_data_distill.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--keep-thinking", action="store_true",            help="保留 <think> block（默认去除）")
    parser.add_argument("--min-turns",     type=int, default=2,            help="最少 tool call 轮数")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        sys.exit(f"[ERROR] runs-dir 不存在: {runs_dir}")

    traj_files = sorted(runs_dir.glob("*/trajectories/*.json"))
    # 排除 condense session 文件（_condense.json 后缀）以及子 agent 文件
    main_files = [
        f for f in traj_files
        if not any(tag in f.name for tag in ("_condense", "judge", "verify", "sub", "search"))
    ]
    print(f"[INFO] 找到 {len(main_files)} 条候选轨迹文件（将按 agent_type 过滤）")

    converted = []
    skipped_status = 0
    skipped_struct = 0

    for f in main_files:
        try:
            traj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {f}: {e}")
            continue

        result = convert_trajectory(traj, keep_thinking=args.keep_thinking)
        if result is None:
            if traj.get("status") != "submit_answer_confirmed":
                skipped_status += 1
            else:
                skipped_struct += 1
            continue

        # 过滤轮数太少的
        tool_calls_count = sum(
            1 for m in result["messages"]
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        if tool_calls_count < args.min_turns:
            skipped_struct += 1
            continue

        converted.append(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in converted:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] 转换完成:")
    print(f"  总轨迹:           {len(main_files)}")
    print(f"  未完成 (跳过):    {skipped_status}")
    print(f"  结构不合法 (跳过): {skipped_struct}")
    print(f"  有效样本:         {len(converted)}")
    print(f"  输出:             {out_path}")


if __name__ == "__main__":
    main()
