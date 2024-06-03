import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./work/summarized")
    parser.add_argument("--versions", type=str, default="all", choices=["all", "rectangular"])
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_eval_result",
        choices=["mean_eval_result", "best_mean_tune_result"],
    )
    parser.add_argument("--pattern", type=str, required=True)
    return vars(parser.parse_args())


def main(root, versions, metric, pattern):
    root = Path(root).expanduser()
    assert root.exists() and root.is_dir()

    folders = [fname for fname in os.listdir(root) if pattern in fname]
    print(f"found {len(folders)} folders")
    print(f"metric: {metric}")

    if versions == "all":
        versions = [
            "cifar",
            "caltech101",
            "dtd",
            # "oxford_flowers102",
            "oxford_iiit_pet",
            "svhn",
            "sun397",
            "patch_camelyon",
            "eurosat",
            "resisc45",
            "diabetic_retinopathy",
            "clevr_count",
            "clevr_dist",
            "dmlab",
            "kitti",
            "dsprites_loc",
            "dsprites_ori",
            "smallnorb_azi",
            "smallnorb_ele",
        ]
    elif versions == "rectangular":
        versions = [
            "caltech101",
            "dtd",
            "oxford_flowers102",
            "oxford_iiit_pet",
            "sun397",
            "diabetic_retinopathy",
            "clevr_count",
            "clevr_dist",
            "dmlab",
            "kitti",
        ]
    else:
        raise NotImplementedError(f"unknown versions '{versions}'")

    # print in order
    results = []
    stds = []
    best_lrs = []
    for version in versions:
        version_folders = [fname for fname in folders if version in fname]
        if len(version_folders) == 0:
            print(f"no result for {version} found")
            continue
        if len(version_folders) > 1:
            print(f"multiple result for {version} found")
            continue
        # search for summary yaml
        results_root = root / version_folders[0]
        summary_fnames = [fname for fname in os.listdir(results_root) if "__summarize__" in fname]
        if len(summary_fnames) == 0:
            print(f"no summary found in {results_root.as_posix()}")
            continue
        if len(version_folders) > 1:
            print(f"multiple summaries found in {results_root.as_posix()}")
            continue
        summary_uri = results_root / summary_fnames[0]
        with open(summary_uri) as f:
            summary = yaml.safe_load(f)
        if metric == "mean_eval_result":
            results.append(summary["result"]["mean_eval_result"])
            df = pd.DataFrame(summary["result"]["mean_tune_results"])
            best_lrs.append(df.loc[df["result"].idxmax()]["lr"])
            eval_results = summary["result"]["eval_results"]
            all_results = [eval_results[i]["result"] for i in range(len(eval_results))]
            stds.append(np.std(all_results))
        elif metric == "best_mean_tune_result":
            df = pd.DataFrame(summary["result"]["mean_tune_results"])
            best_row = df.loc[df["result"].idxmax()]
            best_lr = best_row["lr"]
            best_result = best_row["result"]
            results.append(best_result)
            best_lrs.append(best_lr)
        else:
            raise NotImplementedError
    print("accuracies")
    print("\t".join(map(str, results)))
    print("stds")
    print("\t".join(map(str, stds)))
    print("best lrs")
    print("\t".join(best_lrs))


if __name__ == "__main__":
    main(**parse_args())
