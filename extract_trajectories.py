#!/usr/bin/env python3
"""Extract and analyze trajectories from the submission JSONL file."""
import json
import re
import sys

def extract_thinking_blocks(messages):
    """Extract content between <think> and </think> tags from assistant messages."""
    blocks = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Find all <think>...</think> blocks
            thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            blocks.extend(thinks)
    return blocks

def count_assistant_turns(messages):
    """Count how many assistant messages there are."""
    return sum(1 for msg in messages if msg.get("role") == "assistant")

def analyze_trajectory(line_num, data):
    """Analyze a single trajectory."""
    query_id = data.get("query_id", "unknown")
    status = data.get("status", "unknown")
    predicted_answer = data.get("predicted_answer", "")
    messages = data.get("messages", [])
    
    thinking_blocks = extract_thinking_blocks(messages)
    num_turns = count_assistant_turns(messages)
    
    # Count tool calls
    tool_calls_count = 0
    search_calls = 0
    get_doc_calls = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls_count += 1
                if tc.get("function", {}).get("name") == "search":
                    search_calls += 1
                elif tc.get("function", {}).get("name") == "get_document":
                    get_doc_calls += 1
    
    return {
        "line": line_num,
        "query_id": query_id,
        "status": status,
        "predicted_answer": predicted_answer,
        "num_turns": num_turns,
        "thinking_blocks": thinking_blocks,
        "num_thinking_blocks": len(thinking_blocks),
        "tool_calls": tool_calls_count,
        "search_calls": search_calls,
        "get_doc_calls": get_doc_calls,
    }

def main():
    filepath = "/home/u-longyy/nju-nlp-deep-research/runs/submission_20260513_183320.jsonl"
    
    results = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result = analyze_trajectory(i, data)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}", file=sys.stderr)
    
    # === SUMMARY STATISTICS ===
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total trajectories: {len(results)}")
    
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    
    # Turn distribution
    turns = [r["num_turns"] for r in results]
    print(f"\nTurn distribution:")
    print(f"  Min: {min(turns)}, Max: {max(turns)}, Mean: {sum(turns)/len(turns):.1f}")
    from collections import Counter
    turn_counts = Counter(turns)
    for t in sorted(turn_counts):
        print(f"  {t} turns: {turn_counts[t]} trajectories")
    
    # Thinking block distribution
    thinking_counts = [r["num_thinking_blocks"] for r in results]
    print(f"\nThinking blocks per trajectory:")
    print(f"  Min: {min(thinking_counts)}, Max: {max(thinking_counts)}, Mean: {sum(thinking_counts)/len(thinking_counts):.1f}")
    
    # Tool call distribution
    tool_counts = [r["tool_calls"] for r in results]
    search_counts = [r["search_calls"] for r in results]
    doc_counts = [r["get_doc_calls"] for r in results]
    print(f"\nTool calls per trajectory:")
    print(f"  Total: Min: {min(tool_counts)}, Max: {max(tool_counts)}, Mean: {sum(tool_counts)/len(tool_counts):.1f}")
    print(f"  Search: Min: {min(search_counts)}, Max: {max(search_counts)}, Mean: {sum(search_counts)/len(search_counts):.1f}")
    print(f"  Get doc: Min: {min(doc_counts)}, Max: {max(doc_counts)}, Mean: {sum(doc_counts)/len(doc_counts):.1f}")
    
    # Answers summary
    no_answer = [r for r in results if "cannot be determined" in r["predicted_answer"].lower() or "insufficient" in r["predicted_answer"].lower()]
    print(f"\nTrajectories giving up (cannot determine/insufficient): {len(no_answer)}")
    
    # === PER-TRAJECTORY DETAILS ===
    print("\n" + "=" * 80)
    print("PER-TRAJECTORY DETAILS")
    print("=" * 80)
    for r in results:
        answer_preview = r["predicted_answer"][:120].replace("\n", " ")
        print(f"\nLine {r['line']} | Query {r['query_id']} | Status: {r['status']} | Turns: {r['num_turns']} | Tool calls: {r['tool_calls']} (search: {r['search_calls']}, doc: {r['get_doc_calls']})")
        print(f"  Answer: {answer_preview}...")
        
        # Print thinking blocks summary
        for j, block in enumerate(r["thinking_blocks"]):
            block_preview = block.strip()[:200].replace("\n", " ")
            print(f"  Think #{j+1}: {block_preview}...")
    
    # === THINKING BLOCKS FOR SPECIFIC TRAJECTORIES ===
    target_ids = {"442", "549"}
    print("\n" + "=" * 80)
    print("FULL THINKING BLOCKS FOR TARGET TRAJECTORIES")
    print("=" * 80)
    
    # Add 3 more random-ish ones (different from the two specified)
    other_ids = [r["query_id"] for r in results if r["query_id"] not in target_ids]
    import random
    random.seed(42)
    extra_ids = set(random.sample(other_ids, min(5, len(other_ids))))
    target_ids.update(extra_ids)
    
    for r in results:
        if r["query_id"] in target_ids:
            print(f"\n{'─' * 80}")
            print(f"QUERY ID: {r['query_id']} | Status: {r['status']} | Turns: {r['num_turns']}")
            print(f"PREDICTED ANSWER: {r['predicted_answer'][:300]}")
            print(f"{'─' * 80}")
            for j, block in enumerate(r["thinking_blocks"]):
                print(f"\n─── Thinking Block #{j+1} (Turn {j+1}) ───")
                print(block.strip())
    
    # === ANALYSIS ===
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Pattern analysis in thinking
    breakdown_q = 0
    plan_search = 0
    eval_results = 0
    give_up = 0
    
    for r in results:
        all_thinking = " ".join(r["thinking_blocks"]).lower()
        if any(phrase in all_thinking for phrase in ["break down", "breakdown", "sub-question", "step by step", "first", "second", "third", "finally"]):
            breakdown_q += 1
        if any(phrase in all_thinking for phrase in ["search", "query", "look for", "find"]):
            plan_search += 1
        if any(phrase in all_thinking for phrase in ["not enough", "insufficient", "doesn't match", "doesn't provide", "no direct", "lack"]):
            eval_results += 1
    
    print(f"\nThinking pattern analysis (based on keyword matching):")
    print(f"  Breaks down into sub-questions: {breakdown_q}/{len(results)}")
    print(f"  Plans search strategies: {plan_search}/{len(results)}")
    print(f"  Critically evaluates results: {eval_results}/{len(results)}")
    
    # Common failure patterns
    print(f"\nCommon observations:")
    print(f"  - Many trajectories give up early (1-2 turns) without deep search")
    print(f"  - Common answer pattern: 'cannot be determined from the provided documents'")
    
    # Check if questions have expected answers (pattern: answers that are NOT "cannot determine")
    concrete_answers = [r for r in results if "cannot be determined" not in r["predicted_answer"].lower() and "insufficient" not in r["predicted_answer"].lower()]
    print(f"  - Trajectories with concrete answers (not giving up): {len(concrete_answers)}")
    
    # Save full extracted data to JSON for further analysis
    output_path = "/home/u-longyy/nju-nlp-deep-research/extracted_trajectories.json"
    # Remove large thinking blocks to keep file manageable
    light_results = []
    for r in results:
        lr = {k: v for k, v in r.items() if k != "thinking_blocks"}
        lr["thinking_block_lengths"] = [len(b) for b in r["thinking_blocks"]]
        light_results.append(lr)
    
    with open(output_path, 'w') as f:
        json.dump(light_results, f, indent=2, ensure_ascii=False)
    print(f"\nLight results saved to {output_path}")

if __name__ == "__main__":
    main()
