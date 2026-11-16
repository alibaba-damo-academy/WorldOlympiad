#!/usr/bin/env python3
"""统计 token_usage JSONL 文件中 input_tokens、output_tokens 的总和。"""

import json
import sys
from pathlib import Path


def sum_usage(filepath: str) -> dict:
    total_input = 0
    total_output = 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_input += rec.get("input_tokens", 0)
            total_output += rec.get("output_tokens", 0)
            count += 1
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "call_count": count,
    }


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "logs/token_usage/usage.jsonl"
    )
    if not Path(path).exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    result = sum_usage(path)
    print(f"File: {path}")
    print(f"  total_input_tokens:  {result['total_input_tokens']}")
    print(f"  total_output_tokens: {result['total_output_tokens']}")
    print(f"  total_tokens:        {result['total_tokens']}")
    print(f"  call_count:          {result['call_count']}")


if __name__ == "__main__":
    main()
