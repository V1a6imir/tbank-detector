import os
import cv2
import argparse
from pathlib import Path
from app.inference import TBankDetector
import glob


def iou(boxA, boxB):
    """Вычисление IoU между двумя боксами (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate(weights: str, data_dir: str = "data/yolo/images/val", labels_dir: str = "data/yolo/labels/val"):
    detector = TBankDetector(weights_path=weights)

    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    out_dir = Path("eval/found")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = Path("eval/false_positives")
    fp_dir.mkdir(parents=True, exist_ok=True)

    TP = FP = FN = 0

    for img_path in image_paths:
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, img_name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Пропускаю: не удалось открыть {img_path}")
            continue

        detections = detector.detect(img, conf_threshold=0.5)

        # Загружаем GT боксы
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    h0, w0 = img.shape[:2]
                    x1 = int((x - w / 2) * w0)
                    y1 = int((y - h / 2) * h0)
                    x2 = int((x + w / 2) * w0)
                    y2 = int((y + h / 2) * h0)
                    gt_boxes.append((x1, y1, x2, y2))

        matched = set()
        for (x1, y1, x2, y2, conf) in detections:
            pred_box = (x1, y1, x2, y2)
            found_match = False
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                if iou(pred_box, gt_box) >= 0.5:
                    TP += 1
                    matched.add(j)
                    found_match = True
                    break
            if not found_match:
                FP += 1
                # сохраним FP отдельно
                fp_out = fp_dir / Path(img_path).name
                fp_img = img.copy()
                cv2.rectangle(fp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite(str(fp_out), fp_img)

        FN += len(gt_boxes) - len(matched)

        # Рисуем все найденные боксы
        vis_img = img.copy()
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        save_path = out_dir / Path(img_path).name
        cv2.imwrite(str(save_path), vis_img)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"Images: {len(image_paths)}")
    print(f"TP={TP}  FP={FP}  FN={FN}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1@IoU=0.5 -> {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .onnx model")
    args = parser.parse_args()

    evaluate(weights=args.weights)
