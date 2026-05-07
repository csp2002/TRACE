#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

VALID_EXTS = (".png",)
DEFAULT_SPLITS = ["train", "test"]


def numeric_key(name: str) -> Tuple[int, str]:
    nums = re.findall(r"\d+", name)
    last = int(nums[-1]) if nums else -1
    return (last, name)


def list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        [
            f for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTS) and os.path.isfile(os.path.join(folder, f))
        ],
        key=numeric_key,
    )


def choose_ref_index_v6_5(idx: int, n: int) -> int:
    # exact v6_5 logic
    if idx < n // 2:
        return min(idx + 1, n - 1)
    else:
        return max(idx - 1, 0)


def detect_splits(dataset_dir: str) -> List[str]:
    splits = []
    entries = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    for sp in DEFAULT_SPLITS:
        ct_dir = os.path.join(dataset_dir, sp, "CT")
        if os.path.isdir(ct_dir):
            splits.append(sp)
    for d in entries:
        if d in splits:
            continue
        ct_dir = os.path.join(dataset_dir, d, "CT")
        if os.path.isdir(ct_dir):
            splits.append(d)
    return splits


def process_split_neighbor_strict(root: str, dataset: str, split: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Strict mode:
      - Require every CT slice to have a corresponding mask file with SAME filename under Mask/.
      - Build per-slice neighbor reference mapping using v6_5 rule.
      - If any mask is missing -> raise error (stop immediately).
    """
    split_dir = os.path.join(root, dataset, split)
    ct_root = os.path.join(split_dir, "CT")
    mask_root = os.path.join(split_dir, "Mask")

    if not os.path.isdir(ct_root):
        return {}

    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"[FATAL] Missing Mask folder: {mask_root}")

    mapping: Dict[str, Dict[str, Dict[str, str]]] = {}

    for patient in sorted(os.listdir(ct_root)):
        patient_ct_dir = os.path.join(ct_root, patient)
        patient_mask_dir = os.path.join(mask_root, patient)

        if not os.path.isdir(patient_ct_dir):
            continue
        if not os.path.isdir(patient_mask_dir):
            raise FileNotFoundError(
                f"[FATAL] Missing patient Mask folder.\n"
                f"  CT:   {patient_ct_dir}\n"
                f"  Mask: {patient_mask_dir}"
            )

        ct_files = list_image_files(patient_ct_dir)
        if not ct_files:
            continue

        # --- strict: every CT file must have a mask file ---
        for fname in ct_files:
            mpath = os.path.join(patient_mask_dir, fname)
            if not os.path.exists(mpath):
                raise FileNotFoundError(
                    f"[FATAL] Missing mask for CT slice.\n"
                    f"  Dataset/Split: {dataset}/{split}\n"
                    f"  Patient: {patient}\n"
                    f"  CT:   {os.path.join(patient_ct_dir, fname)}\n"
                    f"  Mask: {mpath}"
                )

        patient_map: Dict[str, Dict[str, str]] = {}
        n = len(ct_files)

        for idx, fname in enumerate(ct_files):
            ct_path = os.path.abspath(os.path.join(patient_ct_dir, fname))
            mask_path = os.path.abspath(os.path.join(patient_mask_dir, fname))

            ref_idx = choose_ref_index_v6_5(idx, n)
            ref_fname = ct_files[ref_idx]
            ref_ct_path = os.path.abspath(os.path.join(patient_ct_dir, ref_fname))
            ref_mask_path = os.path.abspath(os.path.join(patient_mask_dir, ref_fname))

            # --- strict: reference mask must exist too (should always true if above check passed) ---
            if not os.path.exists(ref_mask_path):
                raise FileNotFoundError(
                    f"[FATAL] Missing reference mask (strict).\n"
                    f"  Dataset/Split: {dataset}/{split}\n"
                    f"  Patient: {patient}\n"
                    f"  Current slice: {ct_path}\n"
                    f"  Ref slice:     {ref_ct_path}\n"
                    f"  Ref mask:      {ref_mask_path}"
                )

            patient_map[fname] = {
                "ct_path": ct_path,
                "mask_path": mask_path,
                "ref_filename": ref_fname,
                "ref_ct_path": ref_ct_path,
                "ref_mask_path": ref_mask_path,
            }

        mapping[patient] = patient_map

    return mapping


def write_json(path: str, data: Dict) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def auto_detect_datasets(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build STRICT neighbor-slice JSONs (v6_5 rule).")
    ap.add_argument("--root", type=str, default="./2D_data",
                    help="Base root, e.g., ./2D_data")
    ap.add_argument("--datasets", type=str, default="",
                    help="Comma-separated dataset names (optional). If empty, auto-detect all under --root.")
    ap.add_argument("--outfile", type=str, default="annotation_dict_neighbor.json",
                    help="Output JSON filename to place under each <dataset>/<split>/")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] \
        if args.datasets else auto_detect_datasets(root)

    if not datasets:
        print(f"[WARN] No datasets found under: {root}")
        return

    print(f"[INFO] Datasets to process: {datasets}")

    total_written = 0
    for ds in datasets:
        dataset_dir = os.path.join(root, ds)
        if not os.path.isdir(dataset_dir):
            print(f"[WARN] Skip non-dir dataset: {dataset_dir}")
            continue

        splits = detect_splits(dataset_dir)
        if not splits:
            print(f"[WARN] No valid splits found in {dataset_dir}")
            continue

        for sp in splits:
            mapping = process_split_neighbor_strict(root, ds, sp)
            out_json = os.path.join(root, ds, sp, args.outfile)
            write_json(out_json, mapping)
            print(f"[OK] {ds}/{sp}: wrote {len(mapping)} patients -> {out_json}")
            total_written += 1

    print(f"[DONE] Wrote {total_written} JSON files.")


if __name__ == "__main__":
    main()
