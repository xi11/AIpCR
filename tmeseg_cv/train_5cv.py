import os
import random as rn
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tensorflow.keras.optimizers.legacy import Adam
from transformers import TFAutoModelForSemanticSegmentation

# -----------------------
# Reproducibility
# -----------------------
np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)

# -----------------------
# Paths / config
# -----------------------
fold_dir = Path("/rsrch5/home/trans_mol_path/xpan7/pipelines/artemis/tmeseg_cv/five_folds")  # contains fold_1.csv ... fold_5.csv

img_root  = Path("/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/image")
mask_root = Path("/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng")

image_size   = 512
learning_rate = 1e-4
batch_size    = 8
num_epochs    = 60
auto          = tf.data.AUTOTUNE
n_splits      = 5

# -----------------------
# Labels
# -----------------------
id2label = {
    0: 'background',
    1: 'tumor',
    2: 'stroma',
    3: 'parenchyma',
    4: 'necrosis',
    5: 'fat'
}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# -----------------------
# Model checkpoint
# -----------------------
model_checkpoint = "/rsrch5/home/trans_mol_path/xpan7/tmesegK8/code/model/mit-b3-finetuned-TCGAbcss-e60-lr00001-s512-40x896"

# -----------------------
# Helpers: build image/mask paths from file_name
# -----------------------
def _strip_ext(name: str) -> str:
    # Accept "ABC.png" or "ABC" and return "ABC"
    return Path(name).stem

def make_image_mask_paths(file_names):
    """
    file_name in fold CSV should match your patch basename (with or without .png).
    Builds:
      image: img_root / f"{name}.png"
      mask : mask_root / f"mask_{name}.png"
    """
    stems = [Path(fn).stem for fn in file_names]  # "ABC_001"
    imgs  = [str(img_root / f"{s}.png") for s in stems]
    masks = [str(mask_root / f"mask_{s}.png") for s in stems]
    return imgs, masks, stems

# -----------------------
# TF reading / preprocessing (make train+test consistent)
# -----------------------
def read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.resize(x, [image_size, image_size])
    return x

def read_mask(path):
    y = tf.io.read_file(path)
    y = tf.image.decode_png(y, channels=1)
    # IMPORTANT: nearest for segmentation masks
    y = tf.image.resize(y, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y = tf.cast(y, tf.int32)
    return y

def aug_transforms(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.8, 2.0)
    image = tf.image.random_hue(image, 0.1)
    return image

def preprocess_train(img_path, mask_path):
    image = read_image(img_path)
    mask  = read_mask(mask_path)
    image = aug_transforms(image)
    # Your original code transposes for segformer; keep consistent:
    image = tf.transpose(image, (2, 0, 1))  # (3,H,W)
    mask  = tf.squeeze(mask, axis=-1)       # (H,W)
    return image, mask

def preprocess_test(img_path, mask_path):
    image = read_image(img_path)
    mask  = read_mask(mask_path)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, (2, 0, 1))  # (3,H,W) to match training
    mask  = tf.squeeze(mask, axis=-1)
    return image, mask

def build_ds(img_paths, mask_paths, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if training:
        ds = ds.shuffle(buffer_size=min(len(img_paths), 4096), seed=2023, reshuffle_each_iteration=True)
        ds = ds.map(preprocess_train, num_parallel_calls=auto)
        ds = ds.repeat()
    else:
        ds = ds.map(preprocess_test, num_parallel_calls=auto)
    ds = ds.batch(batch_size).prefetch(auto)
    return ds

# -----------------------
# Dice per patch (per class)
# -----------------------
def dice_per_class_numpy(y_true, y_pred, num_classes: int, eps: float = 1e-7):
    """
    y_true, y_pred: (H,W) int arrays
    returns: (num_classes,) dice for this patch
    """
    dices = np.zeros((num_classes,), dtype=np.float32)
    for c in range(num_classes):
        t = (y_true == c)
        p = (y_pred == c)
        inter = np.logical_and(t, p).sum()
        denom = t.sum() + p.sum()
        dices[c] = (2.0 * inter + eps) / (denom + eps)
    return dices

def predict_mask(model, image_batch, target_hw=(512, 512)):
    """
    image_batch: TF tensor shape (B,3,H,W) following your preprocessing.
    returns predicted labels (B,H,W) as numpy int32
    """
    out = model(image_batch, training=False)
    logits = out.logits  # expected (B, num_labels, h, w)

    # Convert to channel-last for tf.image.resize
    logits = tf.transpose(logits, (0, 2, 3, 1))  # (B, h, w, C)

    # Upsample logits to target size
    logits_up = tf.image.resize(logits, target_hw, method="bilinear")  # (B, H, W, C)

    # Argmax over classes
    pred = tf.argmax(logits_up, axis=-1)  # (B, H, W)
    return pred.numpy().astype(np.int32)

# -----------------------
# Load fold CSVs and CV loop
# -----------------------
# Read all fold files, keep their original columns, add fold column
fold_dfs = []
for k in range(1, n_splits + 1):
    fp = fold_dir / f"fold_{k}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing: {fp}")
    d = pd.read_csv(fp)
    d["fold"] = k
    fold_dfs.append(d)

df_all = pd.concat(fold_dfs, ignore_index=True)

if "file_name" not in df_all.columns:
    raise ValueError("Your fold CSVs must contain a 'file_name' column.")

# Main CV
for test_fold in range(1, n_splits + 1):
    print(f"\n==================== Fold {test_fold}/{n_splits} ====================")

    train_df = df_all[df_all["fold"] != test_fold].reset_index(drop=True)
    test_df  = df_all[df_all["fold"] == test_fold].reset_index(drop=True)

    train_imgs, train_masks, train_names = make_image_mask_paths(train_df["file_name"].tolist())
    test_imgs,  test_masks,  test_names  = make_image_mask_paths(test_df["file_name"].tolist())

    # Optional: compute steps using unique training patches
    num_train = len(train_imgs)
    steps_per_epoch = max(1, num_train // batch_size)

    train_ds = build_ds(train_imgs, train_masks, training=True)
    test_ds  = build_ds(test_imgs,  test_masks,  training=False)

    # Build / compile model (fresh per fold)
    optimizer = Adam(learning_rate=learning_rate)
    model = TFAutoModelForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.compile(optimizer=optimizer)

    # Train
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1
    )

    # Save model per fold
    model_name = Path(model_checkpoint).name
    model_id = f"/rsrch5/home/trans_mol_path/xpan7/pipelines/artemis/tmeseg_cv/model/{model_name}-BRCA-Artemis-CV5-fold{test_fold}-s512-20x512"
    model.save_pretrained(model_id)
    print(f"Saved model: {model_id}")

    # -----------------------
    # Evaluate per patch on test split and save CSV
    # -----------------------
    records = []
    # We need per-patch dice; easiest is to iterate patch-by-patch via batches
    # and keep the corresponding file_name ordering from test_df.
    idx = 0
    for (xb, yb) in test_ds:
        # xb: (B,3,H,W), yb: (B,H,W)
        preds = predict_mask(model, xb)  # (B,H,W)
        ytrue = yb.numpy().astype(np.int32)
        B = preds.shape[0]

        for b in range(B):
            if idx >= len(test_names):
                break
            dvec = dice_per_class_numpy(ytrue[b], preds[b], num_classes=num_labels)
            rec = {
                "fold": test_fold,
                "file_name": test_names[idx]
            }
            for c in range(num_labels):
                rec[f"dice_{c}_{id2label[c]}"] = float(dvec[c])
            rec["dice_mean"] = float(dvec.mean())
            records.append(rec)
            idx += 1

        if idx >= len(test_names):
            break

    perf_df = pd.DataFrame(records)

    out_csv = fold_dir / f"test_patch_dice_fold{test_fold}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(out_csv, index=False)
    print(f"Saved per-patch dice CSV: {out_csv}")

print("\nDone: trained + saved 5 fold models and per-patch dice CSVs.")
