"""Fine-tune a COCO-pretrained YOLO11 detector on RDD2022 India.

Run:  python -m src.train --model yolo11n.pt --epochs 50
      python -m src.train --model yolo11s.pt --epochs 50

Transfer learning: YOLO11 already knows edges, textures and shapes from 80 COCO
classes. We replace its last layer with our 4 damage classes and train the
whole network on road images. The best epoch (by validation mAP) is kept, then
scored once on the held-out test split.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo" / "data.yaml"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def evaluate(weights: Path, name: str) -> dict:
    """Score a trained model on the test split and save the numbers."""
    model = YOLO(weights)
    m = model.val(data=str(DATA), split="test", imgsz=640, batch=16, plots=True,
                  project=str(RUNS), name=f"{name}_test", exist_ok=True, verbose=False)
    names = model.names
    out = {
        "model": name,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class_AP50": {names[int(c)]: round(float(ap), 4) for c, ap in zip(m.box.ap_class_index, m.box.ap50)},
        "ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
        "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    name = Path(args.model).stem
    model = YOLO(args.model)
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=640, batch=args.batch,
        # patience=0: no early stopping. With only 775 validation images, mAP
        # jumps around epoch to epoch (0.32 then 0.24), so a first run with
        # patience=15 stopped at epoch 40 while still improving, and skipped
        # the last 10 epochs where mosaic augmentation is switched off.
        seed=42, deterministic=True, patience=0,
        project=str(RUNS), name=name, exist_ok=True,
        workers=4,  # Windows: more workers mostly adds startup time
    )
    evaluate(RUNS / name / "weights" / "best.pt", name)


if __name__ == "__main__":
    main()
