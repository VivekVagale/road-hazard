# Interview notes: Road Hazard Detection

## Framing

**Object detection, not classification.** A rider needs to know *where* the
pothole is, not just "this frame has damage". Detection gives a box per hazard.

**Dataset: RDD2022, India subset.** 7,706 real Indian road images, taken from
a smartphone mounted in a vehicle, with boxes for road damage. It is public
(CC BY 4.0), from a peer-reviewed benchmark (the CRDDC 2022 challenge), and it
is Indian roads, which look nothing like the US or Japanese roads most models
are trained on.

**Only 4 classes kept.** D00 longitudinal crack, D10 transverse crack,
D20 alligator crack, D40 pothole: the four standard RDD classes. India's extra
labels (D43 crosswalk blur, D44 white-line blur, D50 manhole) are not hazards
a rider avoids, so they were dropped.

**Images with no damage are kept.** 58% of the images have no box. They teach
the model what a normal road looks like, which reduces false alarms.

## The leakage trap (good talking point)

RDD images are frames from drives: image 101 and 102 show the same crack from
almost the same spot. A random split puts one in train and one in test, and
the model "detects" a crack it has literally seen. That inflates mAP.

Fix: split in **blocks of 25 consecutive images**, and shuffle the blocks.
Neighbouring frames stay on the same side of the split. `tests/test_prepare.py`
checks it.

## Model choices

**YOLO11 (Ultralytics).** One-stage detector: one forward pass per image,
real-time on a laptop GPU and exportable to phones (ONNX / TFLite). Two-stage
detectors like Faster R-CNN can be more accurate on small objects but are far
slower, and a dashcam needs real time.

**Transfer learning from COCO.** The backbone already knows edges and
textures from 80 everyday classes; training from scratch on 6k images would
overfit badly.

**Nano vs small.** Nano (2.6M parameters) is the phone-friendly one; small
(9.4M) is more accurate. Training both shows the speed/accuracy trade-off with
real numbers.

**Settings.** 640 px input, batch 16, 50 epochs, early stopping after 15
epochs without improvement, seed 42. Ultralytics' default augmentation (mosaic,
flips, HSV jitter, scaling) matters here because road lighting varies wildly.

## Metrics

**mAP@0.5**: a box counts as correct if it overlaps the true box by at least
50% (IoU >= 0.5); AP is the area under the precision-recall curve per class,
mAP is the mean over classes. **mAP@0.5:0.95** averages over stricter
overlaps; it is lower and rewards tight boxes.

**Per-class results surprised me.** I expected potholes to score highest
(compact blobs, easy to box) and thin cracks lowest. In the first run
alligator cracks scored best (AP50 0.53) and potholes only 0.32. Likely
reasons: alligator cracking covers a large, textured patch that is easy to
box, while Indian potholes vary hugely in size, are often water-filled or in
shadow, and small far-away ones are hard at 640 px. Longitudinal cracks (0.20)
suffer from the thin-box problem: a box around a long crack is mostly road,
and annotators draw very different boxes for the same crack.

**Transverse cracks** have only 57 training boxes (1% of labels) and scored
AP50 0.02. That is a data problem, not a model one.

**Early stopping mistake.** The first run used patience 15 and stopped at
epoch 40 while validation mAP was still rising. With only 775 validation
images, mAP jumps epoch to epoch (0.32, then 0.24), so "no improvement for 15
epochs" fired on noise. It also skipped the final 10 epochs where mosaic
augmentation is turned off, which usually helps. Retrained with early
stopping off; first-run numbers are kept in `results/first_run/`.

## What I would do next

1. Record my own helmet-cam footage on Bengaluru roads and label ~300 frames:
   the real test of whether it transfers to a bike's viewpoint.
2. Add speed breakers (not in RDD) from a separate dataset.
3. Export nano to TFLite and run it on a phone for a live alert.
4. Merge the 3 crack types into one "crack" class if the use case is just
   "rough road ahead"; fewer, better-populated classes.
