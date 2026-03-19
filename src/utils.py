import os
import json
import csv

BASE_DIR = os.getenv("BASE_ARGMIN")
if not BASE_DIR:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_single_json_test_data():
    input_dir = os.path.join(BASE_DIR, "data", "test-data")
    output_path = os.path.join(BASE_DIR, "data", "test_data.json")

    json_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(".json")
    ]
    json_files.sort()

    combined = []
    for filename in json_files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            combined.append(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=True)

def clean_multi_label():
    input_path = os.path.join(BASE_DIR, "data", "education_dimensions_updated.csv")

    hierarchy = {}
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            dimension = (row.get("Dimensions") or "").strip()
            category = (row.get("Categories") or "").strip()
            code = (row.get("CODE") or "").strip()

            if not category or category.lower() == "na":
                continue

            entry = {"category": category, "code": code}
            hierarchy.setdefault(dimension, []).append(entry)

    with open(os.path.join(BASE_DIR, "data", "labels.json"), "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=True)

if __name__ == "__main__":
    # create_single_json_test_data()
    clean_multi_label()
