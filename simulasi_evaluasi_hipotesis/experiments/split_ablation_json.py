# split_ablation_json.py
# Utility script to split ablation_results.json into per-model JSON files
# Used ONLY for analysis & reporting (Reviewer #4), not optimization

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Split ablation_results.json into per-model files")
    ap.add_argument("--in", dest="infile", required=True, help="Path to ablation_results.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    infile = Path(args.infile).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(infile.read_text(encoding="utf-8"))

    if "models" not in data:
        raise ValueError("Invalid ablation_results.json: missing 'models' key")

    for model_name, model_data in data["models"].items():
        outpath = outdir / f"{model_name}.json"
        outpath.write_text(
            json.dumps(model_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {outpath}")

    print("=== Ablation split completed ===")


if __name__ == "__main__":
    main()
