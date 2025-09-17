import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

class TBankDetector:
    def __init__(self, weights_path: str = "models/best.onnx", img_size: int = 640):
        """
        Инициализация детектора
        """
        weights_path = Path(weights_path).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Попробуем сначала CUDA, потом CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(weights_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size = img_size

    def detect(self, image: np.ndarray, conf_threshold=0.25, iou_threshold=0.45):
        """
        Запуск детекции логотипа
        """
        orig_h, orig_w = image.shape[:2]

        # --- preprocess ---
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0

        # --- inference ---
        preds = self.session.run([self.output_name], {self.input_name: img})[0]
        preds = np.squeeze(preds).T  # (N, 8400)

        # --- decode boxes ---
        boxes = preds[:, :4]  # x,y,w,h
        objectness = preds[:, 4]

        if preds.shape[1] > 5:  # есть классы
            class_probs = preds[:, 5:]
            class_conf = class_probs.max(axis=1)
            scores = objectness * class_conf
        else:
            scores = objectness

        # --- фильтрация по conf_threshold ---
        mask = scores > conf_threshold
        boxes, scores = boxes[mask], scores[mask]

        results = []
        if len(scores) > 0:
            # xywh → xyxy
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

            # рескейлим в размер оригинала
            gain_w, gain_h = orig_w / self.img_size, orig_h / self.img_size
            boxes_xyxy[:, [0, 2]] *= gain_w
            boxes_xyxy[:, [1, 3]] *= gain_h

            # --- кастомный NMS ---
            idxs = self._nms(boxes_xyxy, scores, iou_threshold)

            for i in idxs:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
                conf = float(scores[i])
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                if x2 > x1 and y2 > y1:
                    results.append((x1, y1, x2, y2, conf))

        return results

    def _nms(self, boxes, scores, iou_threshold=0.45):
        """
        Реализация NMS на NumPy
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def visualize(self, image: np.ndarray, detections, out_path="debug.jpg"):
        """
        Сохраняет изображение с предсказанными боксами
        detections: список [(x1, y1, x2, y2, conf), ...]
        """
        img = image.copy()
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"tbank_logo {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(out_path, img)
        print(f"[INFO] Visualization saved to {out_path}")
