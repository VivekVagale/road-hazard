"""Convert RDD2022 India (Pascal VOC XML) into a YOLO dataset.

Run:  python -m src.prepare

Source: Arya et al., "RDD2022: A multi-national image dataset for automatic
road damage detection", figshare, CC BY 4.0. Only the India folder is used.

Steps:
  1. unzip India.zip (downloaded by src.download)
  2. read each XML, keep the 4 standard damage classes, convert every box to
     YOLO format: "class x_center y_center width height", all 0-1 fractions
  3. split into train / val / test by *blocks* of consecutive images
"""

import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = ROOT / "data" / "raw" / "RDD2022" / "India.zip"
UNZIP_DIR = ROOT / "data" / "raw" / "India"
OUT = ROOT / "data" / "yolo"

# RDD codes -> our class ids. India also has D43/D44/D50 (crosswalk blur, white
# line blur, manhole); they are not road damage a rider needs to avoid, so skip.
CLASSES = {"D00": "longitudinal_crack", "D10": "transverse_crack", "D20": "alligator_crack", "D40": "pothole"}
CLASS_IDS = {code: i for i, code in enumerate(CLASSES)}

SEED = 42
BLOCK = 25  # consecutive frames from the same drive look almost identical


def unzip() -> Path:
    if not UNZIP_DIR.exists():
        with zipfile.ZipFile(RAW_ZIP) as z:
            z.extractall(UNZIP_DIR)
    return next(UNZIP_DIR.rglob("train"))


def voc_to_yolo(xml_path: Path) -> tuple[list[str], Counter]:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines, counts = [], Counter()
    for obj in root.iter("object"):
        code = obj.find("name").text.strip()
        counts[code] += 1
        if code not in CLASS_IDS:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASS_IDS[code]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, counts


def block_split(stems: list[str]) -> dict[str, list[str]]:
    """80/10/10 split of blocks of consecutive images, not single images.

    A random per-image split would put frame 101 in train and frame 102 in
    test. They show the same crack, so the test score would measure memory,
    not generalisation.
    """
    stems = sorted(stems)
    blocks = [stems[i:i + BLOCK] for i in range(0, len(stems), BLOCK)]
    random.Random(SEED).shuffle(blocks)
    n = len(blocks)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    return {
        "train": [s for b in blocks[:cut1] for s in b],
        "val": [s for b in blocks[cut1:cut2] for s in b],
        "test": [s for b in blocks[cut2:] for s in b],
    }


def main() -> None:
    train_dir = unzip()
    images = {p.stem: p for p in (train_dir / "images").glob("*.jpg")}
    xmls = {p.stem: p for p in (train_dir / "annotations" / "xmls").glob("*.xml")}
    stems = sorted(set(images) & set(xmls))
    print(f"images {len(images)}, annotations {len(xmls)}, paired {len(stems)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    all_counts, kept_counts = Counter(), Counter()
    for split, split_stems in block_split(stems).items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        n_empty = 0
        for s in split_stems:
            lines, counts = voc_to_yolo(xmls[s])
            all_counts += counts
            for line in lines:
                kept_counts[(split, line.split()[0])] += 1
            n_empty += not lines
            shutil.copy2(images[s], OUT / "images" / split / images[s].name)
            (OUT / "labels" / split / f"{s}.txt").write_text("\n".join(lines))
        per_class = {name: kept_counts[(split, str(i))] for i, name in enumerate(CLASSES.values())}
        print(f"{split:5} {len(split_stems):5} images ({n_empty} with no damage)  boxes {per_class}")

    print(f"all raw labels: {dict(all_counts)}")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES.values()))
    )
    print(f"wrote {(OUT / 'data.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
