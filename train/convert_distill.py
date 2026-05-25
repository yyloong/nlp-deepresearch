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
    "status": "submit_answer_confirmed" | "max_turns"
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
  - 只保留 main agent 轨迹（agent_type == "main"）
  - 有 eval_correct.json 时：只保留答对的轨迹（推荐）
  - 无 eval 时：只保留有 predicted_answer 的轨迹
  - 去掉 thinking block（<think>...</think>）以节省训练长度（可选）

注意: status=="max_turns" 的轨迹若答对同样是高质量 SFT 数据，不应排除。
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


def convert_trajectory(
    traj: Dict[str, Any],
    keep_thinking: bool = True,
    correct_ids: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """将一条 trajectory 转换为 SFT 样本。返回 None 表示跳过。"""
    if traj.get("agent_type") != "main":
        return None

    qid = str(traj.get("name", ""))
    if correct_ids is not None:
        # 有 eval 结果：只保留答对的轨迹
        if qid not in correct_ids:
            return None
    else:
        # 无 eval 结果：只保留有答案的轨迹
        if not traj.get("predicted_answer", ""):
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


def load_correct_ids(runs_dir: Path) -> Optional[set]:
    """
    从 runs_dir 下所有 run 目录的 eval_correct.json 汇总答对的 query_id。
    若一个 run 没有 eval_correct.json，则该 run 的轨迹按 predicted_answer 过滤。
    返回 set 或 None（若完全没有 eval 文件）。
    """
    all_ids: set = set()
    found_any = False
    eval_glob = list(runs_dir.glob("*/eval_correct.json")) or list(runs_dir.glob("eval_correct.json"))
    for eval_file in sorted(eval_glob):
        found_any = True
        try:
            with open(eval_file) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        all_ids.add(str(rec["query_id"]))
        except Exception as e:
            print(f"[WARN] 读取 eval_correct.json 失败 {eval_file}: {e}")
    return all_ids if found_any else None


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

    # 收集答对的 query_id
    correct_ids = load_correct_ids(runs_dir)
    if correct_ids is not None:
        print(f"[INFO] 找到 eval_correct.json，共 {len(correct_ids)} 条答对记录 → 只保留答对轨迹")
    else:
        print(f"[INFO] 未找到 eval_correct.json → 按 predicted_answer 过滤（保留所有有答案的轨迹）")

    # 支持两种目录结构:
    #   runs_dir/run_*/trajectories/*.json  (多 run 的父目录)
    #   runs_dir/trajectories/*.json        (单个 run 目录)
    traj_files = sorted(runs_dir.glob("*/trajectories/*.json"))
    if not traj_files:
        traj_files = sorted(runs_dir.glob("trajectories/*.json"))
    # 排除 condense session 文件（_condense.json 后缀）以及子 agent 文件
    main_files = [
        f for f in traj_files
        if not any(tag in f.name for tag in ("_condense", "judge", "verify", "sub", "search"))
    ]
    print(f"[INFO] 找到 {len(main_files)} 条候选轨迹文件")

    converted = []
    skipped_wrong = 0
    skipped_struct = 0

    for f in main_files:
        try:
            traj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {f}: {e}")
            continue

        result = convert_trajectory(traj, keep_thinking=args.keep_thinking, correct_ids=correct_ids)
        if result is None:
            skipped_wrong += 1
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
    print(f"  候选轨迹:         {len(main_files)}")
    filter_label = "答错/无答案 (跳过)" if correct_ids is not None else "无答案 (跳过)"
    print(f"  {filter_label}:   {skipped_wrong}")
    print(f"  结构不合法 (跳过): {skipped_struct}")
    print(f"  有效样本:         {len(converted)}")
    print(f"  输出:             {out_path}")


if __name__ == "__main__":
    main()
