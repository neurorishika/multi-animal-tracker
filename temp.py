import os
import sys
import time

import apriltag
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
from ultralytics import YOLO


def unsharp_mask(image_array, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """Apply an unsharp mask using OpenCV."""
    blurred = cv2.GaussianBlur(image_array, kernel_size, sigma)
    unsharp = cv2.addWeighted(image_array, 1 + amount, blurred, -amount, 0)
    return unsharp


def contrast_enhance(image_array, factor=1.5):
    """Enhance contrast using PIL."""
    pil_img = Image.fromarray(image_array)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(factor)
    return np.array(enhanced)


# ─── CONFIG ────────────────────────────────────────────────────────────────
VIDEO_PATH = sys.argv[1]
# Optional: specify output directory as second argument
# If not provided, output goes to same directory as video
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else None
parent_name = os.path.basename(os.path.dirname(os.path.abspath(VIDEO_PATH)))
# OUTPUT_DIR = os.path.join(output_dir, parent_name)

if OUTPUT_DIR:
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_basename = os.path.basename(VIDEO_PATH).rsplit(".", 1)[0]
    OUT_PKL = os.path.join(OUTPUT_DIR, video_basename + "_apriltagDetect14.pkl")
else:
    OUT_PKL = VIDEO_PATH.rsplit(".", 1)[0] + "_apriltagDetect14.pkl"

CONF_THRESH = 0.1  # confidence threshold for YOLO
PAD_PIXELS = 10  # padding around each detected box
PRINT_PER_FRAME = False  # True → print timing/log per frame; False → only summary
YOLO_WEIGHTS = "detect14.engine"
# '/home/yohann/runs/detect/train17/weights/best.pt'
# '/home/yohann/runs/segment/train14/weights/best.pt'#"best.pt"
# Path to your trained YOLOv8-seg checkpoint

start_time = time.time()
# ─── MODEL INITIALIZATION ──────────────────────────────────────────────────
seg_model = YOLO(YOLO_WEIGHTS)  # your trained YOLOv8-seg checkpoint
# at_detector = apriltag.Detector()                     # AprilTag detector (CPU)
apriltag_params = {
    "threads": 4,  # parallel threads within one detect() call
    "decimate": 1.5,  # typical = 1.0 (no decimation)
    "blur": 0.2,  # sigma for optional blur before decoding
    "refine_edges": 1,  # refine edges for subpixel accuracy
    "decode_sharpening": 9.8,
    "max_hamming": 3,  # allow up to 1-bit error
    "kernel_size": (5, 5),  # for unsharp mask
    "sigma": 1.0,  # for unsharp mask
    "amount": 2.0,  # for unsharp mask
    "contrast_factor": 1.5,  # for contrast enhancement
}

detector = apriltag.apriltag(
    family="tag36ARTag",
    threads=6,  # Each process gets one thread
    maxhamming=apriltag_params["max_hamming"],
    decimate=apriltag_params["decimate"],
    blur=apriltag_params["blur"],
    refine_edges=1,
    qtp_deglitch=1,
    decode_sharpening=apriltag_params["decode_sharpening"],
)


results = seg_model.predict(
    source=VIDEO_PATH,
    conf=CONF_THRESH,
    stream=True,
    batch=2,
    verbose=False,
    half=True,
    task="detect",
    imgsz=1024,
)
results_tag = []
crops_list = []

NUM_FRAMES = 18000

for i, result in enumerate(
    tqdm(results, desc="Processing frames", total=NUM_FRAMES, miniters=20)
):
    # for i, result in enumerate(results):
    # if i > 100:
    #     break
    frame = result.orig_img
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    frame_idx = i
    frame_dict = {("frame", ""): frame_idx}
    # print(f"Processing frame: {frame_idx}")

    # ─── Step 2: CROPPING (CPU) ─────────────────────────────────────────────

    # Collect crops in a list (so we can run AprilTag on them next)
    crops = []  # list[ndarray]
    offsets_xy = []  # list[(x_off, y_off)]
    bbox_xyxy = []  # list[(x1,y1,x2,y2)]  (for later)
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - PAD_PIXELS)
        y1 = max(0, y1 - PAD_PIXELS)
        x2 = min(frame_width, x2 + PAD_PIXELS)
        y2 = min(frame_height, y2 + PAD_PIXELS)

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
        offsets_xy.append((x1, y1))
        bbox_xyxy.append((x1, y1, x2, y2))

    if not crops:
        results_tag.append(frame_dict)
        continue

    # ---- 2. pack crops into a single wide strip ----------------------------
    # (simplest layout: place them left-to-right in order of creation)
    strip_height = max(c.shape[0] for c in crops)
    strip_width = sum(c.shape[1] for c in crops)
    composite_rgb = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)

    x_cursor = 0
    canvas_x_offsets = []  # same order as 'crops'
    for c in crops:
        h, w = c.shape[:2]
        composite_rgb[0:h, x_cursor : x_cursor + w] = c
        canvas_x_offsets.append(x_cursor)  # where this crop starts in canvas
        x_cursor += w

    composite_gray = cv2.cvtColor(composite_rgb, cv2.COLOR_BGR2GRAY)
    sharp = unsharp_mask(
        composite_gray,
        kernel_size=apriltag_params["kernel_size"],
        sigma=apriltag_params["sigma"],
        amount=apriltag_params["amount"],
    )
    enhanced = contrast_enhance(sharp, factor=apriltag_params["contrast_factor"])

    # ---- 3. run apriltag ONCE on the composite -----------------------------
    tags = detector.detect(enhanced)

    # ---- 4. re-project each tag to original frame --------------------------
    for tag in tags:
        cx, cy = tag["center"]  # centre in composite coords
        # figure out which crop it landed in
        # (they are laid out sequentially, so find the segment)
        crop_idx = np.searchsorted(
            np.cumsum([c.shape[1] for c in crops]), cx, side="right"
        )
        crop_x0 = canvas_x_offsets[crop_idx]
        x_off, y_off = offsets_xy[crop_idx]  # top-left of that crop in frame

        # relative position inside the crop
        rel_x = cx - crop_x0
        rel_y = cy  # strip is aligned at y = 0

        # absolute frame coordinates
        abs_x = x_off + rel_x
        abs_y = y_off + rel_y

        tag_id = tag["id"]
        if tag_id > 237:  # ignore unwanted family IDs
            continue

        # convert the 4-corner array the same way
        abs_corners = tag["lb-rb-rt-lt"].copy()
        abs_corners[:, 0] += x_off - crop_x0  # x
        abs_corners[:, 1] += y_off  # y

        frame_dict[(tag_id, "center_x")] = abs_x
        frame_dict[(tag_id, "center_y")] = abs_y
        frame_dict[(tag_id, "corners")] = (
            abs_corners  # Adjust corners to absolute coordinates
        )

    results_tag.append(frame_dict)
results.close()
# Save results to CSV
df = pd.DataFrame(results_tag)
df.columns = pd.MultiIndex.from_tuples(df.columns)
df = df.set_index("frame")

# Now frame_data is a MultiIndex DataFrame
df.to_pickle(OUT_PKL)
# print(f"Processing completed. Results saved to {OUT_PKL}")
total_time = time.time() - start_time
# total time in minutes
print(f"Total processing time for {VIDEO_PATH}: {total_time / 60:.2f} minutes")
