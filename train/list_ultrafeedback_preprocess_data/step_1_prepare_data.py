import argparse
import json
import math
import os
from collections import Counter

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare PRO-format train/dev/test files from a HuggingFace ranking dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NDCG-alignment/ListUltraFeedback",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split_strategy",
        choices=["auto", "ratio"],
        default="auto",
        help=(
            "auto: keep existing test split when available, and split train into train/dev if needed. "
            "ratio: merge all rows then split by train/dev/test ratios."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--dev_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument(
        "--dev_ratio_from_train",
        type=float,
        default=0.05,
        help="Only used by auto strategy when train exists but dev split does not.",
    )
    parser.add_argument("--ranking_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join("..", "..", "data"),
    )
    parser.add_argument("--train_dir", type=str, default="list_ultrafeedback_train_len8")
    parser.add_argument("--dev_dir", type=str, default="list_ultrafeedback_dev_len8")
    parser.add_argument("--test_dir", type=str, default="list_ultrafeedback_test_len8")
    parser.add_argument("--train_file_name", type=str, default="train.json")
    parser.add_argument("--dev_file_name", type=str, default="dev.json")
    parser.add_argument("--test_file_name", type=str, default="test.json")
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="For quick debugging only.",
    )
    return parser.parse_args()


def _format_prefix(prompt):
    prompt = str(prompt).strip()
    return "### User:\n{}\n\n### Assistant:\n".format(prompt)


def _build_pro_sample(raw_sample, source_split, dataset_name, ranking_len):
    if "prompt" not in raw_sample:
        return None, "missing_prompt"
    if "completions" not in raw_sample:
        return None, "missing_completions"
    if "overall_scores" not in raw_sample:
        return None, "missing_scores"

    prompt = raw_sample["prompt"]
    completions = raw_sample["completions"]
    overall_scores = raw_sample["overall_scores"]
    if not isinstance(completions, list) or not isinstance(overall_scores, list):
        return None, "invalid_list_fields"

    scored_candidates = []
    for index, (completion, score) in enumerate(zip(completions, overall_scores)):
        if not isinstance(completion, str):
            continue
        completion = completion.strip()
        if completion == "":
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isnan(score):
            continue
        scored_candidates.append((index, score, completion))

    if len(scored_candidates) < ranking_len:
        return None, "insufficient_candidates"

    scored_candidates = sorted(scored_candidates, key=lambda x: (-x[1], x[0]))
    scored_candidates = scored_candidates[:ranking_len]
    prefix = _format_prefix(prompt)

    return {
        "meta": {
            "dataset": dataset_name,
            "source_split": source_split,
            "source_id": raw_sample.get("id", None),
        },
        "prefix": [prefix for _ in range(ranking_len)],
        "suffix": [candidate[2] for candidate in scored_candidates],
        "reward": [candidate[1] for candidate in scored_candidates],
        "sft_index": 0,
    }, None


def _split_dataset(dataset_dict, args):
    splits = list(dataset_dict.keys())
    print("Available splits:", splits)

    if args.split_strategy == "ratio":
        merged = concatenate_datasets([dataset_dict[name] for name in splits])
        first_split = merged.train_test_split(
            test_size=args.test_ratio,
            seed=args.seed,
            shuffle=True,
        )
        temp_train = first_split["train"]
        test_set = first_split["test"]
        dev_fraction = args.dev_ratio / (args.train_ratio + args.dev_ratio)
        second_split = temp_train.train_test_split(
            test_size=dev_fraction,
            seed=args.seed,
            shuffle=True,
        )
        return second_split["train"], second_split["test"], test_set

    if "train" in dataset_dict:
        train_set = dataset_dict["train"]
        dev_set = None
        for name in ["validation", "valid", "dev"]:
            if name in dataset_dict:
                dev_set = dataset_dict[name]
                break
        test_set = dataset_dict["test"] if "test" in dataset_dict else None

        if dev_set is None:
            train_dev = train_set.train_test_split(
                test_size=args.dev_ratio_from_train,
                seed=args.seed,
                shuffle=True,
            )
            train_set = train_dev["train"]
            dev_set = train_dev["test"]

        if test_set is None:
            dev_test = dev_set.train_test_split(
                test_size=0.5,
                seed=args.seed,
                shuffle=True,
            )
            dev_set = dev_test["train"]
            test_set = dev_test["test"]

        return train_set, dev_set, test_set

    merged = concatenate_datasets([dataset_dict[name] for name in splits])
    first_split = merged.train_test_split(
        test_size=args.test_ratio,
        seed=args.seed,
        shuffle=True,
    )
    temp_train = first_split["train"]
    test_set = first_split["test"]
    dev_fraction = args.dev_ratio / (args.train_ratio + args.dev_ratio)
    second_split = temp_train.train_test_split(
        test_size=dev_fraction,
        seed=args.seed,
        shuffle=True,
    )
    return second_split["train"], second_split["test"], test_set


def _to_pro_samples(dataset_split, split_name, args):
    total = len(dataset_split)
    limit = total if args.max_samples_per_split is None else min(total, args.max_samples_per_split)
    keep = []
    dropped_stats = Counter()

    for index, raw_sample in enumerate(tqdm(dataset_split, total=limit, desc="Converting {}".format(split_name))):
        if index >= limit:
            break
        converted, error = _build_pro_sample(
            raw_sample=raw_sample,
            source_split=split_name,
            dataset_name=args.dataset_name,
            ranking_len=args.ranking_len,
        )
        if converted is None:
            dropped_stats[error] += 1
            continue
        keep.append(converted)
    return keep, dropped_stats


def _write_jsonl(samples, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    ratio_sum = args.train_ratio + args.dev_ratio + args.test_ratio
    assert abs(ratio_sum - 1.0) < 1e-8, "train/dev/test ratios must sum to 1.0"
    assert args.ranking_len >= 2, "ranking_len should be at least 2 for PRO training."

    print("Loading dataset:", args.dataset_name)
    dataset_dict = load_dataset(args.dataset_name, args.dataset_config)
    train_set, dev_set, test_set = _split_dataset(dataset_dict, args)
    print("Split sizes - train/dev/test:", len(train_set), len(dev_set), len(test_set))

    train_samples, train_drop = _to_pro_samples(train_set, "train", args)
    dev_samples, dev_drop = _to_pro_samples(dev_set, "dev", args)
    test_samples, test_drop = _to_pro_samples(test_set, "test", args)

    output_root = os.path.abspath(args.output_root)
    train_file = os.path.join(output_root, args.train_dir, args.train_file_name)
    dev_file = os.path.join(output_root, args.dev_dir, args.dev_file_name)
    test_file = os.path.join(output_root, args.test_dir, args.test_file_name)

    _write_jsonl(train_samples, train_file)
    _write_jsonl(dev_samples, dev_file)
    _write_jsonl(test_samples, test_file)

    print("Saved train file:", train_file, "count=", len(train_samples))
    print("Saved dev file:", dev_file, "count=", len(dev_samples))
    print("Saved test file:", test_file, "count=", len(test_samples))
    print("Dropped stats (train/dev/test):", dict(train_drop), dict(dev_drop), dict(test_drop))


if __name__ == "__main__":
    main()
