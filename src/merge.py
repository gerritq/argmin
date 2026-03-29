import argparse
import glob
import json
import os
from typing import Dict, List, Set, Tuple

BASE_DIR = os.getenv("BASE_ARGMIN")

def unique_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_low_level_codes(label_row: Dict) -> List[str]:
    labels = label_row.get("labels", {})
    low_level = labels.get("low_level", {})
    codes: List[str] = []
    if isinstance(low_level, dict):
        for _, categories in low_level.items():
            if not isinstance(categories, list):
                continue
            for entry in categories:
                if isinstance(entry, dict):
                    code = entry.get("code")
                    if isinstance(code, str) and code:
                        codes.append(code)
    return unique_preserve_order(codes)


def build_code_index(jsonl_dir: str) -> Dict[Tuple[str, int], List[str]]:
    index: Dict[Tuple[str, int], List[str]] = {}
    pattern = os.path.join(BASE_DIR, jsonl_dir, "agentic_labels_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text_id = row.get("TEXT_ID")
                para_number = row.get("para_number")
                if not isinstance(text_id, str) or not isinstance(para_number, int):
                    continue
                codes = extract_low_level_codes(row)
                index[(text_id, para_number)] = codes
    return index


def merge_folder(index: Dict[Tuple[str, int], List[str]], src_dir: str, dst_dir: str) -> None:
    os.makedirs(os.path.join(BASE_DIR, dst_dir), exist_ok=True)
    for name in sorted(os.listdir(os.path.join(BASE_DIR, src_dir))):
        if not name.endswith(".json"):
            continue

        src_path = os.path.join(BASE_DIR, src_dir, name)
        dst_path = os.path.join(BASE_DIR, dst_dir, name)
        text_id = os.path.splitext(name)[0]

        with open(src_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        paragraphs = doc.get("body", {}).get("paragraphs", [])
        if isinstance(paragraphs, list):
            for para in paragraphs:
                if not isinstance(para, dict):
                    continue
                para_number = para.get("para_number")
                if not isinstance(para_number, int):
                    continue
                para["tags"] = index.get((text_id, para_number), [])

        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", default="out_run_1")
    parser.add_argument(
        "--source-dir",
        default="data/test-data-solved-pre-op-and-relationship",
    )
    parser.add_argument("--output-dir", default="data/final")
    args = parser.parse_args()

    index = build_code_index(args.jsonl_dir)
    merge_folder(index, args.source_dir, args.output_dir)
    print(
        f"Merged {len(index)} (TEXT_ID, para_number) label rows into files from "
        f"{args.source_dir} and wrote output to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
