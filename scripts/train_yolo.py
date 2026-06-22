from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "datasets" / "cctv_yolo" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"


def main():
    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset config not found: {DATASET_YAML}")

    train_images = BASE_DIR / "datasets" / "cctv_yolo" / "images" / "train"
    train_labels = BASE_DIR / "datasets" / "cctv_yolo" / "labels" / "train"
    if not any(train_images.glob("*")) or not any(train_labels.glob("*.txt")):
        raise SystemExit(
            "No YOLO training data found. Add images to datasets/cctv_yolo/images/train "
            "and YOLO label .txt files to datasets/cctv_yolo/labels/train."
        )

    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=80,
        imgsz=640,
        project=str(BASE_DIR / "runs"),
        name="cctv_yolo",
        exist_ok=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy2(best_model, MODELS_DIR / "cctv_best.pt")
        print(f"Copied trained model to {MODELS_DIR / 'cctv_best.pt'}")
    else:
        print(f"Training finished, but best.pt was not found under {results.save_dir}")


if __name__ == "__main__":
    main()
