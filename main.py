import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/test_data.json")
    parser.add_argument("--labels", default="data/labels.json")
    parser.add_argument("--output", default="data/agentic_labels.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke-test", type=int, default=0)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    assert args.smoke_test in (0, 1), "smoke-test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)

    label_hierarchy = load_json(args.labels)
    test_data = load_json(args.test_data)

    labeler = AgenticLabeler(args.model_id)
    results: List[Dict[str, Any]] = []

    items = load_paragraphs(test_data)
    if args.smoke_test:
        items = items[:1]
    elif args.limit is not None:
        items = items[: args.limit]
    elif args.n is not None:
        items = items[: args.n]

    for item in tqdm(items, desc="Labeling paragraphs"):
        labels = labeler.label_paragraph(item["paragraph"], label_hierarchy)
        results.append(
            {
                "TEXT_ID": item["TEXT_ID"],
                "TITLE": item["TITLE"],
                "para_number": item["para_number"],
                "paragraph": item["paragraph"],
                "labels": labels,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
