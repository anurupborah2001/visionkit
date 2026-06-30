"""
Simple Classifier for Teachable Machine .h5 models
Works well with TensorFlow 2.15 / 2.16 on Apple Silicon
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class Classifier:
    def __init__(self, model_path: str, labels_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        print(f"Loading model: {model_path}")
        self.model = load_model(model_path, compile=False)

        with open(labels_path, encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Model loaded | TF {tf.__version__} | {len(self.labels)} labels")

        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (224, 224))
        array = np.asarray(resized, dtype=np.float32)
        return (array / 127.0) - 1.0

    def predict(self, img: np.ndarray) -> tuple[list[float], int, str]:
        processed = self.preprocess(img)
        self.data[0] = processed

        predictions = self.model.predict(self.data, verbose=0)
        probs = predictions[0].tolist()

        index = int(np.argmax(predictions))
        label = self.labels[index] if index < len(self.labels) else f"Class {index}"

        return probs, index, label

    def getPrediction(
        self,
        img: np.ndarray,
        draw: bool = True,
        pos: tuple[int, int] = (30, 50),
        scale: float = 1.5,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> tuple[list[float], int]:
        probs, index, label = self.predict(img)

        if draw:
            cv2.putText(
                img, label, pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, thickness
            )

        return probs, index

    def get_label(self, index: int) -> str:
        return (
            self.labels[index]
            if 0 <= index < len(self.labels)
            else f"Unknown ({index})"
        )

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def get_confidence(self, probs: list, index: int) -> float:
        """Return the confidence percentage for a specific class index.

        Args:
          probs: Probability list from predict().
          index: Class index to query.
        Returns:
          float: 0.0–100.0
        """
        if not probs or index >= len(probs):
            return 0.0
        return probs[index] * 100.0

    def predict_top_n(self, img: np.ndarray, n: int = 3):
        """Return the top-N predictions sorted by descending confidence.

        Args:
          img: BGR numpy array.
          n: Number of top predictions to return.
        Returns:
          List[dict]: [{'label': str, 'index': int, 'confidence': float}, ...]
        """
        probs, _, _ = self.predict(img)
        indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:n]
        return [
            {"label": self.get_label(i), "index": i, "confidence": probs[i] * 100.0}
            for i in indices
        ]

    def get_all_predictions(self, probs: list):
        """Return all class predictions paired with their labels.

        Args:
          probs: Probability list from predict().
        Returns:
          List[dict]: [{'label': str, 'index': int, 'confidence': float}] sorted desc.
        """
        return sorted(
            [
                {"label": self.get_label(i), "index": i, "confidence": p * 100.0}
                for i, p in enumerate(probs)
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

    def is_confident(self, probs: list, threshold: float = 70.0) -> bool:
        """Return True if the top prediction confidence meets the threshold.

        Args:
          probs: Probability list from predict().
          threshold: Minimum confidence percentage (default 70 %).
        Returns:
          bool
        """
        if not probs:
            return False
        return max(probs) * 100.0 >= threshold

    def predict_batch(self, images: list):
        """Run predict() on a list of images and return all results.

        Args:
          images: List of BGR numpy arrays.
        Returns:
          List[dict]: [{'label': str, 'index': int, 'confidence': float}]
        """
        results = []
        for img in images:
            probs, index, label = self.predict(img)
            results.append(
                {
                    "label": label,
                    "index": index,
                    "confidence": probs[index] * 100.0,
                }
            )
        return results


# ====================== Quick Test ======================

if __name__ == "__main__":
    MODEL_PATH = "hand-gesture/hand-sign-detection/model/keras_model.h5"
    LABELS_PATH = "hand-gesture/hand-sign-detection/model/labels.txt"

    classifier = Classifier(MODEL_PATH, LABELS_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened. Try changing to cv2.VideoCapture(1)")
        exit()

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        probs, idx = classifier.getPrediction(
            frame, draw=True, scale=1.7, color=(0, 255, 100)
        )
        conf = probs[idx] * 100
        print(f"→ {classifier.get_label(idx)} | Confidence: {conf:.1f}%")

        cv2.imshow("Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
