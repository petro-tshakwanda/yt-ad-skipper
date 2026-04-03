#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import tensorflow as tf

def load_npz_files(features_dir):
    paths = sorted(pathlib.Path(features_dir).glob("*_segments.npz"))
    return paths

def dataset_from_npz(npz_paths, batch_size=2):
    def gen():
        for p in npz_paths:
            d = np.load(p)
            yield d["z"].astype("float32"), d["y"].astype("float32")
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    )
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([None, None], [None]),
        padding_values=(0.0, 0.0)
    )
    return ds

def build_model(input_dim):
    inputs = tf.keras.Input(shape=(None, input_dim), name="segments")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    outputs = tf.keras.layers.Activation("sigmoid")(logits)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data_features")
    parser.add_argument("--out-dir", type=str, default="models/tf_model")
    parser.add_argument("--batch-size", type=int, default=2)  # ✅ CORRECT
    # parser_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    npz_paths = load_npz_files(args.features_dir)
    if not npz_paths:
        raise SystemExit("[ERROR] No feature files found.")

    # peek one file to get input_dim
    d = np.load(npz_paths[0])
    input_dim = d["z"].shape[1]
    print(f"[INFO] Input dim: {input_dim}")

    ds = dataset_from_npz(npz_paths, batch_size=args.batch_size)
    model = build_model(input_dim)
    model.summary()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ FIXED: Add .keras extension
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best.keras"),
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # simple split: first 10% as val
    n = len(npz_paths)
    split = max(1, int(0.1 * n))
    train_paths = npz_paths[split:]
    val_paths = npz_paths[:split]
    train_ds = dataset_from_npz(train_paths, batch_size=args.batch_size)
    val_ds = dataset_from_npz(val_paths, batch_size=args.batch_size)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt_cb]
    )

    # ✅ FIXED: Add .keras extension
    model.save(out_dir / "final.keras")
    print(f"[INFO] Saved model to {out_dir}")

if __name__ == "__main__":
    main()
