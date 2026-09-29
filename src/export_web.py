"""Export a trained detector to ONNX for the browser demo in web/.

Run:  python -m src.export_web --model yolo11n

ONNX is a portable model format; onnxruntime-web runs it in the browser with
WebAssembly, so the demo needs no server. The nano model is used because it
is 10 MB and fast enough on a phone.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n")
    name = ap.parse_args().model
    weights = ROOT / "runs" / name / "weights" / "best.pt"
    onnx = YOLO(weights).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, device="cpu")
    dest = ROOT / "web" / "models" / "yolo11n.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, dest)
    print(f"{dest.relative_to(ROOT)}  {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
