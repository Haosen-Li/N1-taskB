"""Download only the optional WikiText test data; evaluation loads models as needed."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "data" / "wikitext103_test.jsonl")
    a = p.parse_args()
    from datasets import load_dataset
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ds.to_json(str(a.out))
    print(a.out)


if __name__ == "__main__":
    main()
