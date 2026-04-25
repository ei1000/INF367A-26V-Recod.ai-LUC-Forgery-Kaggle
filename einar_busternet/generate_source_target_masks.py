from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset_utils import list_labeled_samples, load_image_from_path, load_union_mask_from_paths
from einar_busternet.source_target_masks import derive_source_target_masks_from_arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BusterNet source/target masks from Kaggle union masks.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--diff-threshold", type=float, default=5.0)
    parser.add_argument("--component-change-fraction", type=float, default=0.25)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def generate_source_target_masks(
    *,
    data_root: Path = Path("data"),
    diff_threshold: float = 5.0,
    component_change_fraction: float = 0.25,
    min_component_area: int = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    data_root = Path(data_root)
    source_dir = data_root / "train_masks_source"
    target_dir = data_root / "train_masks_target"
    metadata_path = data_root / "train_masks_source_target_metadata.csv"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    forged_samples = [sample for sample in list_labeled_samples(data_root) if sample.label == "forged"]
    rows = []

    for sample in tqdm(forged_samples, desc="source/target masks"):
        source_path = source_dir / f"{sample.case_id}.npy"
        target_path = target_dir / f"{sample.case_id}.npy"
        if source_path.exists() and target_path.exists() and not overwrite:
            rows.append(
                {
                    "case_id": sample.case_id,
                    "status": "skipped_existing",
                    "source_path": source_path.as_posix(),
                    "target_path": target_path.as_posix(),
                }
            )
            continue

        authentic_path = data_root / "train_images" / "authentic" / f"{sample.case_id}.png"
        authentic = load_image_from_path(authentic_path) if authentic_path.exists() else None
        forged = load_image_from_path(sample.image_path)
        union_mask = load_union_mask_from_paths(sample.mask_paths)
        masks = derive_source_target_masks_from_arrays(
            authentic,
            forged,
            union_mask,
            diff_threshold=diff_threshold,
            component_change_fraction=component_change_fraction,
            min_component_area=min_component_area,
        )

        np.save(source_path, masks.source_mask.astype(np.uint8))
        np.save(target_path, masks.target_mask.astype(np.uint8))

        component_scores = masks.component_scores
        target_components = int(component_scores["is_target"].sum()) if "is_target" in component_scores else 0
        source_components = int((~component_scores["is_target"] & ~component_scores["is_tiny"]).sum()) if "is_target" in component_scores else 0
        rows.append(
            {
                "case_id": sample.case_id,
                "status": masks.status,
                "has_authentic_pair": authentic is not None,
                "union_pixels": int(masks.union_mask.sum()),
                "target_pixels": int(masks.target_mask.sum()),
                "source_pixels": int(masks.source_mask.sum()),
                "component_count": int(len(component_scores)),
                "target_component_count": target_components,
                "source_component_count": source_components,
                "diff_threshold": diff_threshold,
                "component_change_fraction": component_change_fraction,
                "min_component_area": min_component_area,
                "source_path": source_path.as_posix(),
                "target_path": target_path.as_posix(),
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_path, index=False)
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    metadata = generate_source_target_masks(
        data_root=args.data_root,
        diff_threshold=args.diff_threshold,
        component_change_fraction=args.component_change_fraction,
        min_component_area=args.min_component_area,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(metadata)} rows to {args.data_root / 'train_masks_source_target_metadata.csv'}")
    if "status" in metadata:
        print(metadata["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
