import argparse
import json
import os
from typing import Any, Dict, List
import sys
from src.agents import AgenticLabeler
from tqdm import tqdm

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paragraphs(test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for doc in test_data:
        text_id = doc.get("TEXT_ID")
        title = doc.get("TITLE")
        paragraphs = doc.get("body", {}).get("paragraphs", [])
        for item in paragraphs:
            para_number = item.get("para_number")
            para_text = item.get("para_en", "")
            items.append({
                "TEXT_ID": text_id,
                "TITLE": title,
                "para_number": para_number,
                "paragraph": para_text,
            })
    return items


def save_jsonl_chunk(output_path: str, chunk: List[Dict[str, Any]], start_idx: int, end_idx: int) -> None:
    base, _ = os.path.splitext(output_path)
    chunk_path = f"{base}_{start_idx}_{end_idx}.jsonl"
    os.makedirs(os.path.dirname(chunk_path) or ".", exist_ok=True)
    with open(chunk_path, "w", encoding="utf-8") as f:
        for row in chunk:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/test_data.json")
    parser.add_argument("--labels", default="data/labels.json")
    parser.add_argument("--output", default="out/agentic_labels.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke-test", type=int, default=0)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--low-level-mode", choices=["debate", "single"], default="debate")
    args = parser.parse_args()

    assert args.smoke_test in (0, 1), "smoke-test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)

    label_hierarchy = load_json(args.labels)
    test_data = load_json(args.test_data)

    labeler = AgenticLabeler(args.model_id, low_level_mode=args.low_level_mode)
    results: List[Dict[str, Any]] = []

    items = load_paragraphs(test_data)
    if args.smoke_test:
        items = items[:1]
    elif args.limit is not None:
        items = items[: args.limit]
    elif args.n is not None:
        items = items[: args.n]

    chunk_size = 100
    chunk_rows: List[Dict[str, Any]] = []
    chunk_start_idx = 0

    for idx, item in enumerate(tqdm(items, desc="Labeling paragraphs")):
        labels = labeler.label_paragraph(item["paragraph"], label_hierarchy)
        row = {
            "TEXT_ID": item["TEXT_ID"],
            "TITLE": item["TITLE"],
            "para_number": item["para_number"],
            "paragraph": item["paragraph"],
            "labels": labels,
        }
        results.append(row)
        chunk_rows.append(row)

        if len(chunk_rows) == chunk_size:
            save_jsonl_chunk(args.output, chunk_rows, chunk_start_idx, idx)
            chunk_rows = []
            chunk_start_idx = idx + 1

    if chunk_rows:
        save_jsonl_chunk(args.output, chunk_rows, chunk_start_idx, len(items) - 1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
