"""Convert VisDrone DET annotations to the COCO format used by this project."""

import argparse
import json
import struct
from pathlib import Path


SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}

CATEGORY_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}


def read_image_size(path):
    """Read JPEG/PNG dimensions without requiring Pillow."""
    with path.open("rb") as image:
        header = image.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])
        if not header.startswith(b"\xff\xd8"):
            raise ValueError(f"Unsupported image format: {path}")

        image.seek(2)
        while True:
            byte = image.read(1)
            while byte == b"\xff":
                byte = image.read(1)
            if not byte:
                break

            marker = byte[0]
            if marker in JPEG_SOF_MARKERS:
                segment = image.read(7)
                if len(segment) != 7:
                    break
                _, _, height, width = struct.unpack(">HBHH", segment)
                return width, height

            if marker == 0xD9:
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue

            length_bytes = image.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            image.seek(segment_length - 2, 1)

    raise ValueError(f"Could not read image dimensions: {path}")


def parse_annotation(path, image_id, width, height, next_annotation_id):
    annotations = []
    annotation_id = next_annotation_id

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.strip().rstrip(",").split(",")
        if len(values) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 columns, got {len(values)}")

        x, y, box_width, box_height, score, category_id = map(float, values[:6])
        category_id = int(category_id)

        # VisDrone categories 0 and 11 are ignored regions/others. Scores of
        # zero also mark boxes that must not be used as training targets.
        if score <= 0 or not 1 <= category_id <= len(CATEGORY_NAMES):
            continue

        x_min = max(0.0, x)
        y_min = max(0.0, y)
        x_max = min(float(width), x + box_width)
        y_max = min(float(height), y + box_height)
        clipped_width = x_max - x_min
        clipped_height = y_max - y_min
        if clipped_width <= 0 or clipped_height <= 0:
            continue

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id - 1,
            "bbox": [x_min, y_min, clipped_width, clipped_height],
            "area": clipped_width * clipped_height,
            "iscrowd": 0,
            "segmentation": [],
        })
        annotation_id += 1

    return annotations, annotation_id


def convert_split(split_dir, output_path, allow_missing_annotations=False):
    image_dir = split_dir / "images"
    annotation_dir = split_dir / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} must contain both images/ and annotations/ directories"
        )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    annotation_stems = {path.stem for path in annotation_dir.glob("*.txt")}
    missing = [path for path in image_paths if path.stem not in annotation_stems]
    if missing and not allow_missing_annotations:
        examples = ", ".join(path.name for path in missing[:5])
        raise RuntimeError(
            f"{split_dir.name}: {len(missing)} images have no annotation file "
            f"(examples: {examples}). Re-extract the annotations or pass "
            "--allow-missing-annotations to convert only matched images."
        )
    if missing:
        print(
            f"WARNING: {split_dir.name}: skipping {len(missing)} images "
            "without annotation files"
        )

    paired_images = [path for path in image_paths if path.stem in annotation_stems]
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(CATEGORY_NAMES)
        ],
    }

    annotation_id = 1
    for image_id, image_path in enumerate(paired_images, start=1):
        width, height = read_image_size(image_path)
        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })
        annotations, annotation_id = parse_annotation(
            annotation_dir / f"{image_path.stem}.txt",
            image_id,
            width,
            height,
            annotation_id,
        )
        coco["annotations"].extend(annotations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, separators=(",", ":")))
    print(
        f"{split_dir.name}: wrote {len(coco['images'])} images and "
        f"{len(coco['annotations'])} annotations to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET TXT annotations to COCO JSON."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("configs/dataset/VisDrone"),
        help="Directory containing the VisDrone2019-DET split directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_DIRS,
        default=["train", "val"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/dataset/visdrone_annotations"),
        help="Directory in which train.json, val.json, and test-dev.json are written.",
    )
    parser.add_argument(
        "--allow-missing-annotations",
        action="store_true",
        help="Skip images whose matching TXT file is missing.",
    )
    args = parser.parse_args()

    for split in args.splits:
        split_dir = args.root / SPLIT_DIRS[split]
        convert_split(
            split_dir,
            args.output_dir / f"{split}.json",
            allow_missing_annotations=args.allow_missing_annotations,
        )


if __name__ == "__main__":
    main()
