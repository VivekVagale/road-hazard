"""Road Hazard Detection demo.  Run:  streamlit run app.py"""

import random
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TEST_IMAGES = ROOT / "data" / "yolo" / "images" / "test"

st.set_page_config(page_title="Road Hazard Detection", page_icon="🛣️", layout="wide")


def trained_models() -> dict[str, Path]:
    return {p.parent.parent.name: p for p in sorted(RUNS.glob("*/weights/best.pt"))}


@st.cache_resource
def load(path: str) -> YOLO:
    return YOLO(path)


st.title("🛣️ Road Hazard Detection")
st.caption("YOLO11 fine-tuned on 6,000 Indian road images (RDD2022). Finds potholes and three kinds of cracks.")

models = trained_models()
if not models:
    st.error("No trained model yet. Run `python -m src.prepare` then `python -m src.train`.")
    st.stop()

with st.sidebar:
    name = st.selectbox("Model", list(models), index=len(models) - 1)
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05,
                     help="Lower finds more damage but adds false boxes")
    source = st.radio("Input", ["Random test image", "Upload image", "Upload video"])

model = load(str(models[name]))


def show_counts(result) -> None:
    names = result.names
    counts = {}
    for c in result.boxes.cls.tolist():
        counts[names[int(c)]] = counts.get(names[int(c)], 0) + 1
    if not counts:
        st.info("No damage found above the threshold.")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        st.write(f"**{k.replace('_', ' ')}**: {v}")


if source == "Random test image":
    images = sorted(TEST_IMAGES.glob("*.jpg"))
    if st.button("🎲 Another image") or "img" not in st.session_state:
        st.session_state["img"] = str(random.choice(images))
    path = st.session_state["img"]
    st.caption(f"Held-out test image {Path(path).name}: the model never saw it during training.")
    r = model.predict(path, conf=conf, verbose=False)[0]
    left, right = st.columns([3, 1])
    left.image(r.plot()[:, :, ::-1], use_container_width=True)
    with right:
        show_counts(r)

elif source == "Upload image":
    up = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"])
    if up:
        r = model.predict(Image.open(up).convert("RGB"), conf=conf, verbose=False)[0]
        left, right = st.columns([3, 1])
        left.image(r.plot()[:, :, ::-1], use_container_width=True)
        with right:
            show_counts(r)

else:
    up = st.file_uploader("Dashcam / helmet-cam clip (keep it short)", type=["mp4", "mov", "avi"])
    if up:
        src = Path(tempfile.mkdtemp()) / up.name
        src.write_bytes(up.read())
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = src.with_name("annotated.mp4")
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        bar = st.progress(0.0, text="Detecting...")
        n = 0
        for r in model.predict(str(src), conf=conf, stream=True, verbose=False):
            out.write(r.plot())
            n += 1
            bar.progress(min(n / max(total, 1), 1.0), text=f"frame {n}/{total}")
        out.release()
        st.video(str(out_path))
