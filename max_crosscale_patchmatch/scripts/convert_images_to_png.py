import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all .tif/.tiff/.jpg/.jpeg images in a folder to .png."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder to scan recursively for images to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional root output folder. If omitted, a sibling folder with '_png' appended is created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .png files if they already exist.",
    )
    return parser.parse_args()


def iter_source_images(input_dir: Path):
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def resolve_output_path(source_path: Path, input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return source_path.with_suffix(".png")

    relative_path = source_path.relative_to(input_dir).with_suffix(".png")
    return output_dir / relative_path


def convert_image(source_path: Path, target_path: Path, overwrite: bool) -> bool:
    if target_path.exists() and not overwrite:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        image.save(target_path, format="PNG")

    return True


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for source_path in iter_source_images(input_dir):
        target_path = resolve_output_path(source_path, input_dir, output_dir)
        if convert_image(source_path, target_path, args.overwrite):
            converted += 1
            print(f"Converted: {source_path} -> {target_path}")
        else:
            skipped += 1
            print(f"Skipped:   {target_path}")

    print(f"Done. Converted {converted} file(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
