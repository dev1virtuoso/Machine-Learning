import json
import glob
import logging
import os
from pathlib import Path

from transformers import BertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PREPARE] - %(message)s",
)

def prepare_2019_dataset(input_dir: str, output_file: str, max_samples=None):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processed = 0

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f_out:
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            logging.error(f"No JSON files found in {input_dir}")
            return

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    raw_text = data.get("text") or data.get("content") or ""

                    if not raw_text or len(raw_text) < 10:
                        continue
                    tokens = tokenizer.encode(
                        raw_text,
                        truncation=True,
                        max_length=128,
                        add_special_tokens=True,
                    )

                    alpha_ratio = sum(c.isalpha() for c in raw_text) / len(raw_text)
                    logic_score = 0.0 if alpha_ratio > 0.7 else 1.0

                    payload = {
                        "text": raw_text,
                        "label_ids": tokens,
                        "logic_score": logic_score,
                    }
                    f_out.write(json.dumps(payload) + "\n")
                    processed += 1

                    if max_samples and processed >= max_samples:
                        break
            except Exception as e:
                logging.warning(f"Skipping {file_path} due to error: {e}")

    logging.info(f"Successfully prepared {processed} samples in {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw JSON files",
    )
    parser.add_argument("--output", type=str, default="train.jsonl", help="Output jsonl file")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to process"
    )
    args = parser.parse_args()
    prepare_2019_dataset(args.input_dir, args.output, args.max_samples)
