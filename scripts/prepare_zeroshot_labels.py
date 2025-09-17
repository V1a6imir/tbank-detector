import os, json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
from transformers import (
    Owlv2Processor, Owlv2ForObjectDetection,
    OwlViTProcessor, OwlViTForObjectDetection,
    CLIPProcessor, CLIPModel
)

# --- ПАРАМЕТРЫ ---
RAW_DIR = Path("data/raw")
OUT_YOLO_IMG = Path("data/yolo/images")
OUT_YOLO_LBL = Path("data/yolo/labels")
VIS_DIR = Path("data/zeroshot_vis")
NEG_DIR = Path("negatives")

SPLIT_VAL_RATIO = 0.2
OWL_THRESHOLD = 0.30        # поджали порог
CLIP_MARGIN = 0.15
CLIP_ACCEPT = 0.65          # CLIP-доверие для принятия бокса

# Тексты-запросы
TBANK_QUERIES = [
    "T Bank logo", "T-Bank logo", "logo of T Bank",
    "логотип Т-банк", "логотип Т банк", "эмблема Т-банк"
]
TINKOFF_QUERIES = [
    "Tinkoff logo", "logo of Tinkoff",
    "логотип Тинькофф", "эмблема Тинькофф"
]


def load_owl_model():
    try:
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        model.eval()
        return processor, model, "owlv2"
    except Exception:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        model.eval()
        return processor, model, "owlvit"


def run_owl(processor, model, image: Image.Image, queries: List[str], backend: str, threshold: float):
    inputs = processor(
        text=queries,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].cpu().numpy().astype(np.float32)
    scores = results["scores"].cpu().numpy().astype(np.float32)
    labels = results["labels"].cpu().numpy().astype(np.int32)
    return boxes, scores, labels


def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_proc, clip_model


def clip_filter(clip_proc, clip_model, image_crops: List[Image.Image]) -> List[float]:
    if len(image_crops) == 0:
        return []
    texts = ["T-Bank logo", "Tinkoff logo"]
    inputs = clip_proc(text=texts, images=image_crops, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = clip_model(**inputs)
    logits = out.logits_per_image
    probs = logits.softmax(dim=1).cpu().numpy()
    return probs[:, 0].tolist()  # вероятность класса T-Bank


def to_yolo_line(x1, y1, x2, y2, W, H, cls_id=0) -> str:
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def draw_vis(img_np, boxes, colors, path):
    img = img_np.copy()
    for (x1, y1, x2, y2), color in zip(boxes, colors):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.imwrite(str(path), img)


def is_valid_box(x1, y1, x2, y2, W, H,
                 min_rel_area=0.0005, max_rel_area=0.35,
                 min_ar=0.4, max_ar=2.5):
    """
    Проверка бокса:
    - относительная площадь: от 0.05% до 35% картинки
    - aspect ratio: от 0.4 до 2.5 (разрешаем прямоугольники, не только квадраты)
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    area = w * h
    rel_area = area / (W * H)
    ar = w / (h + 1e-6)
    return (min_rel_area <= rel_area <= max_rel_area) and (min_ar <= ar <= max_ar)



def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_YOLO_IMG.mkdir(parents=True, exist_ok=True)
    OUT_YOLO_LBL.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    processor, owl_model, backend = load_owl_model()
    clip_proc, clip_model = load_clip()

    images = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    accepted = 0
    tinkoff_count = 0
    no_logo = 0

    for p in tqdm(images, desc="Zero-shot labeling"):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        tb_boxes, _, _ = run_owl(processor, owl_model, img, TBANK_QUERIES, backend, OWL_THRESHOLD)
        tk_boxes, _, _ = run_owl(processor, owl_model, img, TINKOFF_QUERIES, backend, OWL_THRESHOLD)

        all_boxes, all_tags = [], []
        for b in tb_boxes: all_boxes.append(b); all_tags.append("tb")
        for b in tk_boxes: all_boxes.append(b); all_tags.append("tk")

        crops = []
        for (x1, y1, x2, y2) in all_boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x2 <= x1 or y2 <= y1:
                crops.append(None)
                continue
            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)

        probs_tb = clip_filter(clip_proc, clip_model, [c for c in crops if c is not None])
        probs_iter = iter(probs_tb)
        probs_full = [next(probs_iter) if c is not None else 0.0 for c in crops]

        keep_boxes = []
        neg_tag = False
        for (box, tag, p_tb) in zip(all_boxes, all_tags, probs_full):
            x1, y1, x2, y2 = map(int, box)
            if not is_valid_box(x1, y1, x2, y2, W, H):
                continue

            # принимаем бокс как T-Bank, если CLIP уверен ≥ 0.65
            if p_tb >= CLIP_ACCEPT:
                keep_boxes.append([x1, y1, x2, y2])

            # явно Тинькофф
            if tag == "tk" and p_tb < 0.35:
                neg_tag = True
        if len(keep_boxes) == 0:
            no_logo += 1
            if neg_tag:
                tinkoff_count += 1
                cv2.imwrite(str(NEG_DIR / p.name), img_np)
            continue

        lbl_path = OUT_YOLO_LBL / f"{p.stem}.txt"
        img_out_path = OUT_YOLO_IMG / p.name
        with open(lbl_path, "w", encoding="utf-8") as f:
            for (x1, y1, x2, y2) in keep_boxes:
                f.write(to_yolo_line(x1, y1, x2, y2, W, H, cls_id=0) + "\n")
        img.save(img_out_path)

        draw_vis(img_np, keep_boxes, colors=[(0, 255, 0)] * len(keep_boxes), path=VIS_DIR / p.name)
        accepted += 1

    print(f"Accepted with T-Bank boxes: {accepted}")
    print(f"Moved to negatives (Tinkoff/hard neg): {tinkoff_count}")
    print(f"No logo detected: {no_logo}")

    imgs = sorted([p for p in OUT_YOLO_IMG.iterdir() if p.is_file()])
    n_val = max(1, int(len(imgs) * SPLIT_VAL_RATIO))
    val = set([p.name for p in imgs[-n_val:]])

    def move_pair(pimg: Path, split: str):
        plbl = OUT_YOLO_LBL / (pimg.stem + ".txt")
        (OUT_YOLO_IMG / split).mkdir(parents=True, exist_ok=True)
        (OUT_YOLO_LBL / split).mkdir(parents=True, exist_ok=True)
        pimg.rename(OUT_YOLO_IMG / split / pimg.name)
        if plbl.exists():
            plbl.rename(OUT_YOLO_LBL / split / plbl.name)

    for p in imgs:
        split = "val" if p.name in val else "train"
        move_pair(p, split)

    data_yaml = {
        "path": "data/yolo",
        "train": "images/train",
        "val": "images/val",
        "names": ["tbank_logo"]
    }
    with open("data/yolo/data.yaml", "w", encoding="utf-8") as f:
        f.write(json.dumps(data_yaml, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
