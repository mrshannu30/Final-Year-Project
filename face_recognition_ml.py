import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC


@dataclass
class TrainingExample:
    encoding: np.ndarray
    label: str


def load_training_data(train_dir: str) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for person_name in sorted(os.listdir(train_dir)):
        person_dir = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print(f"Skipping {image_path}: no face detected.")
                continue
            encodings = face_recognition.face_encodings(image, face_locations)
            for encoding in encodings:
                examples.append(TrainingExample(encoding=encoding, label=person_name))
    return examples


def train_model(examples: List[TrainingExample]) -> SVC:
    if not examples:
        raise ValueError("No training examples found. Add images to the train directory.")
    X = np.array([ex.encoding for ex in examples])
    y = np.array([ex.label for ex in examples])
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)
    return model


def save_model(model: SVC, output_path: str) -> None:
    with open(output_path, "wb") as file:
        pickle.dump(model, file)


def load_model(model_path: str) -> SVC:
    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict_faces(
    model: SVC,
    image_path: str,
    threshold: float = 0.6,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    image = face_recognition.load_image_file(image_path)
    rgb_image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return []
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    results = []
    for encoding, (top, right, bottom, left) in zip(encodings, face_locations):
        probabilities = model.predict_proba([encoding])[0]
        best_idx = int(np.argmax(probabilities))
        best_label = model.classes_[best_idx]
        confidence = float(probabilities[best_idx])
        label = best_label if confidence >= threshold else "Unknown"
        results.append((label, confidence, (top, right, bottom, left)))
    return results


def annotate_image(
    image_path: str,
    predictions: List[Tuple[str, float, Tuple[int, int, int, int]]],
    output_path: str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    for label, confidence, (top, right, bottom, left) in predictions:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(
            image,
            text,
            (left, max(top - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(output_path, image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face recognition with ML (SVM).")
    parser.add_argument("--train-dir", help="Directory with subfolders per person.")
    parser.add_argument("--model-out", default="face_model.pkl", help="Model output path.")
    parser.add_argument("--predict", help="Image path for prediction.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold.")
    parser.add_argument("--annotate-out", help="Output path for annotated image.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.train_dir:
        examples = load_training_data(args.train_dir)
        model = train_model(examples)
        save_model(model, args.model_out)
        print(f"Model saved to {args.model_out}.")

    if args.predict:
        model = load_model(args.model_out)
        predictions = predict_faces(model, args.predict, args.threshold)
        if not predictions:
            print("No faces detected.")
        else:
            for label, confidence, _ in predictions:
                print(f"Prediction: {label} ({confidence:.2f})")
        if args.annotate_out:
            annotate_image(args.predict, predictions, args.annotate_out)
            print(f"Annotated image saved to {args.annotate_out}.")

    if not args.train_dir and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()
