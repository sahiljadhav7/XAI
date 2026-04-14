import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CANONICAL_CLASSES = ["Lung Opacity", "Normal", "Viral Pneumonia"]
CLASS_ALIASES = {
    "Lung Opacity": {"lungopacity", "lung_opacity", "lung opacity"},
    "Normal": {"normal"},
    "Viral Pneumonia": {"viralpneumonia", "viral_pneumonia", "viral pneumonia"},
}


def normalize_name(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_class_directories(data_dir):
    data_dir = Path(data_dir)
    available = {}

    for child in data_dir.iterdir():
        if not child.is_dir():
            continue

        normalized = normalize_name(child.name)
        for canonical_name, aliases in CLASS_ALIASES.items():
            if normalized in {normalize_name(alias) for alias in aliases}:
                available[canonical_name] = child

    missing = [class_name for class_name in CANONICAL_CLASSES if class_name not in available]
    if missing:
        raise FileNotFoundError(
            "Missing class folders in data directory. Expected folders for: "
            + ", ".join(missing)
        )

    return available


def gather_image_paths(class_directories):
    image_paths = []
    labels = []

    for index, class_name in enumerate(CANONICAL_CLASSES):
        class_dir = class_directories[class_name]
        class_images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not class_images:
            raise FileNotFoundError(f"No images found for class '{class_name}' in {class_dir}")

        image_paths.extend([str(path) for path in class_images])
        labels.extend([index] * len(class_images))

    return image_paths, labels


def decode_image(path, label, image_size):
    image_bytes = tf.io.read_file(path)
    image_tensor = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image_tensor = tf.image.resize(image_tensor, [image_size, image_size])
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = preprocess_input(image_tensor)
    return image_tensor, tf.one_hot(label, len(CANONICAL_CLASSES))


def build_dataset(paths, labels, image_size, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: decode_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            lambda image, label: (
                tf.image.random_flip_left_right(image),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(image_size, learning_rate, dropout):
    base_model = MobileNet(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CANONICAL_CLASSES), activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the lung X-ray model expected by this project."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing class folders for Lung Opacity, Normal, and Viral Pneumonia.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/models/lung_disease",
        help="Where to save model.h5 and class metadata.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.h5"
    class_names_path = output_dir / "class_names.json"
    metrics_path = output_dir / "metrics.json"

    class_directories = find_class_directories(args.data_dir)
    image_paths, labels = gather_image_paths(class_directories)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=args.val_size,
        stratify=labels,
        random_state=args.seed,
    )

    train_dataset = build_dataset(
        train_paths,
        train_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=True,
    )
    val_dataset = build_dataset(
        val_paths,
        val_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
    )

    model = build_model(
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    evaluation = model.evaluate(val_dataset, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(value) for value in evaluation]))
    metrics["classes"] = CANONICAL_CLASSES
    metrics["train_samples"] = len(train_paths)
    metrics["val_samples"] = len(val_paths)
    metrics["last_conv_layer"] = "conv_pw_13_relu"

    class_names_path.write_text(json.dumps(CANONICAL_CLASSES, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved class names to: {class_names_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("Training history keys:", list(history.history.keys()))


if __name__ == "__main__":
    main()
