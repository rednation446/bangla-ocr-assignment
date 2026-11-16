#!/usr/bin/env python3
"""Train a CRNN + CTC word recogniser for handwritten Bangla and track it in MLflow.

BanglaLekha-Isolated only has single letters. This script builds whole words
from them on the fly: it picks real handwritten letters for each position,
draws the vowel signs (kar) that the dataset lacks as pen strokes, joins the
letters with a headline (matra) and distorts the result. A CNN reads the strip
column by column, two BiLSTM layers add context, and the CTC loss teaches the
network to output the letter sequence without any character segmentation.

Examples
    python train.py                                   # run 1, default settings
    python train.py --epochs 20 --learning-rate 5e-4 --run-name crnn-slow-lr
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from keras import layers
from PIL import Image, ImageDraw, ImageFilter

import ocr_core as core

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_LABEL_LEN = 24
NO_HEADLINE = {"এ", "ঐ", "ও", "ঔ", "ঙ", "ঞ", "ৎ", "ং", "ঃ"}


# ======================================================================= data
def find_class_folders(root: Path) -> dict[int, Path]:
    """Find the numbered BanglaLekha folders (1..50) anywhere under root."""
    found: dict[int, Path] = {}
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            if name.isdigit() and 1 <= int(name) <= 50 and int(name) not in found:
                found[int(name)] = Path(dirpath) / name
    return found


def load_glyph(path: Path, height: int = 48) -> np.ndarray | None:
    try:
        gray = np.asarray(Image.open(path).convert("L"))
    except OSError:
        return None
    gray = core.white_on_black(gray)
    crop = core.crop_ink(gray, threshold=80, pad=1)
    if crop is None or min(crop.shape) < 6:
        return None
    h, w = crop.shape
    new_w = int(np.clip(round(w * height / h), 10, int(height * 1.7)))
    return np.asarray(Image.fromarray(crop).resize((new_w, height), Image.Resampling.LANCZOS))


def load_banglalekha(root: Path, per_class: int, val_share: float, seed: int):
    folders = find_class_folders(root)
    missing = [i for i in range(1, 50) if i not in folders]
    if missing:
        raise SystemExit(
            f"Could not find BanglaLekha class folders {missing[:10]} under '{root}'.\n"
            "Download BanglaLekha-Isolated from https://data.mendeley.com/datasets/hf6sf8zrkc/2 "
            "and unzip it inside the data/ folder (see README)."
        )
    rng = random.Random(seed)
    train_bank: dict[str, list[np.ndarray]] = {}
    val_bank: dict[str, list[np.ndarray]] = {}
    for number in range(1, 50):                     # 50 = chandrabindu, not used in words
        char = core.FOLDER_TO_CHAR[str(number)]
        files = sorted(p for p in folders[number].iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        rng.shuffle(files)
        files = files[:per_class]
        glyphs = [g for g in (load_glyph(p) for p in files) if g is not None]
        n_val = max(2, int(len(glyphs) * val_share))
        val_bank[char], train_bank[char] = glyphs[:n_val], glyphs[n_val:]
        print(f"  folder {number:2d} {char}: {len(glyphs)} images", end="\r")
    print()
    return train_bank, val_bank, folders


# ============================================================ word synthesis
def spline(points: list[tuple[float, float]], per_segment: int = 10) -> list[tuple[float, float]]:
    """Catmull-Rom curve through the control points: smooth, pen-like strokes."""
    pts = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = (np.array(p) for p in pts[i - 1:i + 3])
        for t in np.linspace(0, 1, per_segment, endpoint=False):
            out.append(tuple(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
    out.append(points[-1])
    return out


class Pen:
    def __init__(self, draw: ImageDraw.ImageDraw, width: int, rng: np.random.Generator, size: float):
        self.draw, self.width, self.rng, self.size = draw, width, rng, size

    def stroke(self, points: list[tuple[float, float]]) -> None:
        jitter = 0.035 * self.size
        pts = [(x + self.rng.normal(0, jitter), y + self.rng.normal(0, jitter)) for x, y in points]
        curve = spline(pts) if len(pts) > 2 else pts
        self.draw.line(curve, fill=255, width=self.width, joint="curve")
        r = self.width / 2
        for x, y in (curve[0], curve[-1]):
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)


def draw_sign(pen: Pen, sign: str, box: dict, prev: dict | None, nxt: dict | None,
              top: float, base: float) -> None:
    """Draw one vowel sign as a handwritten stroke next to / under its letter."""
    h = base - top
    x0, x1 = box["x0"], box["x1"]
    xm = (x0 + x1) / 2
    if sign == "া":
        pen.stroke([(xm, top), (xm + 0.02 * h, (top + base) / 2), (xm, base)])
        pen.stroke([(x0 - 0.1 * h, top), (x1, top)])
    elif sign == "ি":
        reach = (nxt["x1"] - nxt["x0"]) if nxt else h * 0.8
        pen.stroke([(xm, base), (xm - 0.02 * h, (top + base) / 2), (xm, top)])
        pen.stroke([(xm, top), (xm + 0.25 * reach, top - 0.32 * h), (xm + 0.75 * reach, top - 0.22 * h),
                    (xm + 0.9 * reach, top + 0.05 * h)])
    elif sign == "ী":
        reach = (prev["x1"] - prev["x0"]) if prev else h * 0.8
        pen.stroke([(xm, top), (xm + 0.02 * h, (top + base) / 2), (xm, base)])
        pen.stroke([(xm, top), (xm - 0.25 * reach, top - 0.32 * h), (xm - 0.75 * reach, top - 0.22 * h),
                    (xm - 0.9 * reach, top + 0.05 * h)])
    elif sign in ("ে", "ৈ"):
        w = x1 - x0
        pen.stroke([(x1, top), (x0 + 0.25 * w, top + 0.2 * h), (x0 + 0.2 * w, top + 0.45 * h),
                    (x0 + 0.7 * w, top + 0.5 * h), (x0 + 0.55 * w, top + 0.65 * h)])
        pen.stroke([(x0, top), (x1 + 0.1 * h, top)])
        if sign == "ৈ":
            pen.stroke([(x0 + 0.35 * w, top), (x0 + 0.15 * w, top - 0.25 * h),
                        (x0 + 0.55 * w, top - 0.32 * h), (x0 + 0.7 * w, top - 0.05 * h)])
    elif sign in core.BELOW_KARS and prev is not None:
        bw = prev["x1"] - prev["x0"]
        cx = prev["x0"] + bw * pen.rng.uniform(0.45, 0.7)
        y = base - 0.02 * h
        if sign == "ু":
            pts = [(cx, y), (cx - 0.25 * bw, y + 0.2 * h), (cx + 0.05 * bw, y + 0.34 * h),
                   (cx + 0.3 * bw, y + 0.18 * h)]
        elif sign == "ূ":
            pts = [(cx, y), (cx + 0.25 * bw, y + 0.15 * h), (cx, y + 0.32 * h),
                   (cx - 0.22 * bw, y + 0.2 * h), (cx + 0.05 * bw, y + 0.1 * h), (cx + 0.32 * bw, y + 0.34 * h)]
        else:  # ৃ
            pts = [(cx - 0.12 * bw, y), (cx + 0.18 * bw, y + 0.12 * h), (cx - 0.1 * bw, y + 0.24 * h),
                   (cx + 0.22 * bw, y + 0.32 * h)]
        pen.stroke(pts)


SIGN_WIDTH = {"া": 0.28, "ি": 0.32, "ী": 0.32, "ে": 0.42, "ৈ": 0.45}


def render_word(visual: list[str], bank: dict[str, list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    """Compose one word image (white ink on black) from real letters + drawn signs."""
    char_h = int(rng.integers(40, 54))
    top = int(char_h * 0.5)
    base = top + char_h * 0.88
    height = top + char_h + int(char_h * 0.55)
    gap_lo, gap_hi = -int(char_h * 0.08), int(char_h * 0.12)

    # pass 1: layout
    boxes: list[dict] = []
    x = 12
    for tok in visual:
        if tok in core.BELOW_KARS:
            boxes.append({"tok": tok, "x0": x, "x1": x, "kind": "below"})
            continue
        if tok in core.KARS:
            w = int(char_h * SIGN_WIDTH[tok] * rng.uniform(0.85, 1.2))
            boxes.append({"tok": tok, "x0": x, "x1": x + w, "kind": "sign"})
        else:
            glyph = bank[tok][int(rng.integers(len(bank[tok])))]
            gh = int(char_h * rng.uniform(0.88, 1.04))
            gw = max(6, int(glyph.shape[1] * gh / glyph.shape[0] * rng.uniform(0.9, 1.1)))
            glyph = np.asarray(Image.fromarray(glyph).resize((gw, gh), Image.Resampling.BILINEAR))
            boxes.append({"tok": tok, "x0": x, "x1": x + gw, "kind": "base", "glyph": glyph,
                          "y0": top + int(rng.integers(-2, 3))})
            w = gw
        x += w + int(rng.integers(gap_lo, gap_hi + 1))
    width = x + 24
    ink = np.zeros((height, width), dtype=np.uint8)

    # pass 2: real handwritten letters
    for b in boxes:
        if b["kind"] == "base":
            g, y0 = b["glyph"], b["y0"]
            region = ink[y0:y0 + g.shape[0], b["x0"]:b["x0"] + g.shape[1]]
            np.maximum(region, g[:region.shape[0], :region.shape[1]], out=region)

    # pass 3: vowel signs and the headline, drawn with a pen of similar width
    img = Image.fromarray(ink)
    pen = Pen(ImageDraw.Draw(img), max(2, int(round(char_h * rng.uniform(0.055, 0.09)))), rng, char_h)
    bases = [b for b in boxes if b["kind"] == "base"]
    for i, b in enumerate(boxes):
        if b["kind"] == "base":
            continue
        prev = next((c for c in reversed(boxes[:i]) if c["kind"] == "base"), None)
        nxt = next((c for c in boxes[i + 1:] if c["kind"] == "base"), None)
        draw_sign(pen, b["tok"], b, prev, nxt, top + 1, base)
    if bases and rng.random() < 0.6:
        y = top + 1 + rng.normal(0, 1)
        run_start = None
        prev_end = 0
        for b in [c for c in boxes if c["kind"] != "below"] + [None]:
            joins = b is not None and b["tok"] not in NO_HEADLINE
            if joins and run_start is None:
                run_start = b["x0"]
            elif not joins and run_start is not None:
                end = prev_end
                if end - run_start > char_h * 0.6:
                    pen.stroke([(run_start, y), (end, y + rng.normal(0, 1.5))])
                run_start = None
            if b is not None:
                prev_end = b["x1"]
    return augment(img, rng)


def augment(img: Image.Image, rng: np.random.Generator) -> np.ndarray:
    w, h = img.size
    shear = rng.uniform(-0.25, 0.25)
    img = img.transform((w + int(abs(shear) * h), h), Image.Transform.AFFINE,
                        (1, shear, -shear * h if shear > 0 else 0, 0, 1, 0), Image.Resampling.BILINEAR)
    img = img.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BILINEAR, expand=True)
    roll = rng.random()
    if roll < 0.25:
        img = img.filter(ImageFilter.MaxFilter(3))
    elif roll < 0.35:
        img = img.filter(ImageFilter.MinFilter(3))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    strip = core.fit_word(np.asarray(img))
    if strip is None:
        return np.zeros((core.IMG_H, core.IMG_W), dtype=np.uint8)
    if rng.random() < 0.3:
        noise = rng.normal(0, 8, strip.shape)
        strip = np.clip(strip.astype(np.float32) * rng.uniform(0.75, 1.0) + noise, 0, 255).astype(np.uint8)
    return strip


def random_clusters(rng: np.random.Generator) -> list[tuple[str, str | None]]:
    """Plausible but random letter sequences, so the model does not just memorise the dictionary."""
    signs = core.KARS + [core.O_KAR]
    clusters = []
    for i in range(int(rng.integers(1, 6))):
        if i == 0 and rng.random() < 0.2:
            clusters.append((core.VOWELS[int(rng.integers(len(core.VOWELS)))], None))
            continue
        base = core.CONSONANTS[int(rng.integers(len(core.CONSONANTS)))]
        sign = None if rng.random() < 0.45 else signs[int(rng.integers(len(signs)))]
        clusters.append((base, sign))
    if rng.random() < 0.1:
        clusters.append((core.WORD_FINAL_SIGNS[int(rng.integers(3))], None))
    return clusters


def sample_word(rng, lexicon_visual, lexicon_ratio) -> list[str]:
    if lexicon_visual and rng.random() < lexicon_ratio:
        return lexicon_visual[int(rng.integers(len(lexicon_visual)))]
    return core.clusters_to_visual(random_clusters(rng))


def encode(visual: list[str], index: dict[str, int]) -> np.ndarray:
    label = np.zeros(MAX_LABEL_LEN, dtype=np.int32)
    ids = [index[t] for t in visual][:MAX_LABEL_LEN]
    label[:len(ids)] = ids
    return label


class SyntheticWordBatches(keras.utils.PyDataset):
    """Endless supply of freshly drawn word images; every epoch sees new ones."""

    def __init__(self, bank, lexicon_visual, steps, batch_size, lexicon_ratio, seed, **kwargs):
        super().__init__(**kwargs)
        self.bank, self.lexicon_visual = bank, lexicon_visual
        self.steps, self.batch_size, self.lexicon_ratio = steps, batch_size, lexicon_ratio
        self.seed, self.epoch = seed, 0
        self.index = {t: i + 1 for i, t in enumerate(core.TOKENS)}

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.epoch += 1

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        x = np.zeros((self.batch_size, core.IMG_H, core.IMG_W, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, MAX_LABEL_LEN), dtype=np.int32)
        for i in range(self.batch_size):
            visual = sample_word(rng, self.lexicon_visual, self.lexicon_ratio)
            x[i, ..., 0] = render_word(visual, self.bank, rng)
            y[i] = encode(visual, self.index)
        return x, y


def build_validation(bank, lexicon_visual, size, seed):
    rng = np.random.default_rng(seed + 1000)
    index = {t: i + 1 for i, t in enumerate(core.TOKENS)}
    x = np.zeros((size, core.IMG_H, core.IMG_W, 1), dtype=np.float32)
    y = np.zeros((size, MAX_LABEL_LEN), dtype=np.int32)
    words, from_lexicon = [], []
    for i in range(size):
        use_lex = i % 2 == 0 and bool(lexicon_visual)
        visual = sample_word(rng, lexicon_visual, 1.0 if use_lex else 0.0)
        x[i, ..., 0] = render_word(visual, bank, rng)
        y[i] = encode(visual, index)
        words.append(core.visual_to_word(visual))
        from_lexicon.append(use_lex)
    return x, y, words, np.array(from_lexicon)


# ===================================================================== model
def build_crnn(num_classes: int, units: int) -> keras.Model:
    inputs = keras.Input((core.IMG_H, core.IMG_W, 1), name="word_image")
    x = layers.Rescaling(1.0 / 255)(inputs)

    def conv(x, filters, pool=None):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D(pool)(x) if pool else x

    x = conv(x, 32, (2, 2))                    # 32 x 128
    x = conv(x, 64, (2, 2))                    # 16 x 64
    x = conv(x, 96)
    x = conv(x, 96, (2, 1))                    # 8 x 64  (keep horizontal detail)
    x = conv(x, 128)
    x = conv(x, 128, (2, 1))                   # 4 x 64
    x = layers.Permute((2, 1, 3))(x)           # width becomes the time axis
    x = layers.Reshape((core.IMG_W // 4, (core.IMG_H // 16) * 128))(x)
    x = layers.Dense(160, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.2))(x)
    outputs = layers.Dense(num_classes, name="ctc_logits")(x)
    return keras.Model(inputs, outputs, name="bangla_word_crnn")


# ================================================================ evaluation
def evaluate(model, x_val, y_val_words, from_lexicon, decoder: core.WordDecoder) -> dict:
    logits = model.predict(x_val, batch_size=64, verbose=0)
    errors = total = exact = lex_hits = 0
    rows = []
    for i, word in enumerate(y_val_words):
        out = decoder.decode(logits[i], use_lexicon=bool(from_lexicon[i]))
        truth_vis = core.word_to_visual(word)
        raw_vis = core.word_to_visual(out["raw_word"]) or list(out["raw_word"])
        errors += core.edit_distance(raw_vis, truth_vis)
        total += len(truth_vis)
        exact += out["raw_word"] == word
        if from_lexicon[i]:
            lex_hits += out["word"] == word
        rows.append((i, word, out["raw_word"], out["word"]))
    n_lex = max(1, int(from_lexicon.sum()))
    return {"val_cer": errors / max(1, total), "val_word_acc": exact / len(y_val_words),
            "val_word_acc_dictionary": lex_hits / n_lex, "rows": rows}


class TrackingCallback(keras.callbacks.Callback):
    def __init__(self, x_val, words, from_lexicon, decoder):
        super().__init__()
        self.x_val, self.words, self.from_lexicon, self.decoder = x_val, words, from_lexicon, decoder
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_cer": [], "val_word_acc": []}
        self.best_cer, self.best_weights = 9.9, None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        result = evaluate(self.model, self.x_val, self.words, self.from_lexicon, self.decoder)
        for key in ("loss", "val_loss"):
            self.history[key].append(float(logs.get(key, np.nan)))
        self.history["val_cer"].append(result["val_cer"])
        self.history["val_word_acc"].append(result["val_word_acc"])
        mlflow.log_metrics({"loss": float(logs.get("loss", np.nan)),
                            "val_loss": float(logs.get("val_loss", np.nan)),
                            "val_cer": result["val_cer"], "val_word_acc": result["val_word_acc"],
                            "val_word_acc_dictionary": result["val_word_acc_dictionary"],
                            "learning_rate": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))},
                           step=epoch + 1)
        if result["val_cer"] < self.best_cer:
            self.best_cer, self.best_weights = result["val_cer"], self.model.get_weights()
        print(f"  val CER {result['val_cer']:.3f} | word accuracy {result['val_word_acc']:.3f} "
              f"| with dictionary {result['val_word_acc_dictionary']:.3f}")


# ================================================================= artifacts
def plot_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="train CTC loss")
    axes[0].plot(epochs, history["val_loss"], label="val CTC loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["val_cer"], label="val character error rate")
    axes[1].plot(epochs, history["val_word_acc"], label="val word accuracy (no dictionary)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_image_grid(images: np.ndarray, path: Path, cols: int = 3, rows: int = 6, title_prefix="#") -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 1.0))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{title_prefix}{i}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_class_sheet(bank: dict[str, list[np.ndarray]], path: Path) -> None:
    """One sample per BanglaLekha folder, labelled by folder number, to check labels.json."""
    fig, axes = plt.subplots(5, 10, figsize=(12, 6.5))
    for number, ax in zip(range(1, 51), axes.flat):
        ax.axis("off")
        char = core.FOLDER_TO_CHAR[str(number)]
        if bank.get(char):
            ax.imshow(bank[char][0], cmap="gray")
            ax.set_title(f"folder {number}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ====================================================================== main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Bangla handwritten word recogniser.")
    p.add_argument("--data-dir", default="data", help="folder that contains BanglaLekha-Isolated")
    p.add_argument("--per-class", type=int, default=800, help="letter images loaded per class")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steps-per-epoch", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lstm-units", type=int, default=96)
    p.add_argument("--lexicon-ratio", type=float, default=0.5,
                   help="share of training words taken from lexicon.txt (rest are random letter strings)")
    p.add_argument("--val-size", type=int, default=600)
    p.add_argument("--workers", type=int, default=4, help="threads that draw training words")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--always-save", action="store_true",
                   help="overwrite models/model.keras even if an earlier run was better")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    core.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    core.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    core.write_labels()

    print(f"Loading BanglaLekha-Isolated letters from {Path(args.data_dir).resolve()} ...")
    train_bank, val_bank, folders = load_banglalekha(Path(args.data_dir), args.per_class, 0.12, args.seed)
    lexicon = core.read_lexicon()
    lexicon_visual = [core.word_to_visual(w) for w in lexicon]
    print(f"{sum(map(len, train_bank.values()))} training letters, "
          f"{sum(map(len, val_bank.values()))} validation letters, {len(lexicon)} dictionary words")

    train_data = SyntheticWordBatches(train_bank, lexicon_visual, args.steps_per_epoch, args.batch_size,
                                      args.lexicon_ratio, args.seed, workers=args.workers)
    x_val, y_val, val_words, from_lexicon = build_validation(val_bank, lexicon_visual, args.val_size, args.seed)
    decoder = core.WordDecoder(core.TOKENS, lexicon)

    model = build_crnn(len(core.TOKENS) + 1, args.lstm_units)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=5.0), loss=keras.losses.CTC())

    mlflow.set_tracking_uri(core.TRACKING_URI)
    if mlflow.get_experiment_by_name(core.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(core.EXPERIMENT_NAME, artifact_location=core.MLFLOW_ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(core.EXPERIMENT_NAME)
    run_name = args.run_name or f"crnn-ctc-e{args.epochs}-lr{args.learning_rate:g}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "dataset": "BanglaLekha-Isolated (folders 1-49) + drawn vowel signs",
            "model": "CRNN: 6 conv (32-64-96-96-128-128) + 2 BiLSTM, CTC loss",
            "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "lstm_units": args.lstm_units, "lexicon_ratio": args.lexicon_ratio,
            "per_class": args.per_class, "val_size": args.val_size, "seed": args.seed,
            "image_size": f"{core.IMG_H}x{core.IMG_W}", "num_classes": len(core.TOKENS) + 1,
            "dictionary_words": len(lexicon), "parameters": model.count_params(),
        })
        mlflow.set_tag("student", "Redowanul Akbar (1000129290)")

        tracker = TrackingCallback(x_val, val_words, from_lexicon, decoder)
        started = time.time()
        model.fit(train_data, validation_data=(x_val, y_val), epochs=args.epochs, verbose=2,
                  callbacks=[tracker, keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3,
                                                                        min_lr=1e-5, verbose=1)])
        seconds = time.time() - started

        model.set_weights(tracker.best_weights)            # keep the epoch with the lowest CER
        final = evaluate(model, x_val, val_words, from_lexicon, decoder)
        mlflow.log_metrics({"final_val_cer": final["val_cer"], "final_val_word_acc": final["val_word_acc"],
                            "final_val_word_acc_dictionary": final["val_word_acc_dictionary"],
                            "train_seconds": seconds})

        out_dir = core.MLFLOW_DIR / "tmp_run_files"
        out_dir.mkdir(exist_ok=True)
        plot_curves(tracker.history, out_dir / "training_curves.png")
        save_image_grid(x_val[:18], out_dir / "validation_words.png")
        save_class_sheet(train_bank, out_dir / "banglalekha_folders.png")
        with open(out_dir / "validation_predictions.tsv", "w", encoding="utf-8") as f:
            f.write("index\ttruth\tmodel_reading\tfinal_word\n")
            for row in final["rows"]:
                f.write("\t".join(map(str, row)) + "\n")
        run_model = out_dir / "model.keras"
        serving = build_crnn(len(core.TOKENS) + 1, args.lstm_units)   # same graph, no optimizer state
        serving.set_weights(model.get_weights())
        serving.save(run_model)
        for name in ("training_curves.png", "validation_words.png", "banglalekha_folders.png",
                     "validation_predictions.tsv", "model.keras"):
            mlflow.log_artifact(str(out_dir / name), artifact_path="outputs")
        mlflow.log_artifact(str(core.LABELS_PATH), artifact_path="outputs")

        # The app always serves the best run so far (lowest validation CER).
        previous = json.loads(core.MODEL_CARD_PATH.read_text()) if core.MODEL_CARD_PATH.exists() else None
        better = previous is None or final["val_cer"] < previous["val_cer"]
        if better or args.always_save or not core.MODEL_PATH.exists():
            run_model.replace(core.MODEL_PATH)
            core.MODEL_CARD_PATH.write_text(json.dumps({
                "run_id": run.info.run_id, "run_name": run_name, "val_cer": final["val_cer"],
                "val_word_acc": final["val_word_acc"],
                "val_word_acc_dictionary": final["val_word_acc_dictionary"],
                "saved_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
            mlflow.set_tag("served_by_app", "yes")
            print(f"Saved {core.MODEL_PATH.relative_to(core.BASE_DIR)} (this run is now used by app.py)")
        else:
            mlflow.set_tag("served_by_app", "no")
            print(f"Kept the existing model: run '{previous['run_name']}' had a lower CER "
                  f"({previous['val_cer']:.3f}). Use --always-save to overwrite.")
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()

        print(f"\nRun '{run_name}' ({run.info.run_id}) finished in {seconds / 60:.1f} min")
        print(f"  character error rate      {final['val_cer']:.3f}")
        print(f"  word accuracy, free       {final['val_word_acc']:.3f}")
        print(f"  word accuracy, dictionary {final['val_word_acc_dictionary']:.3f}")
        print(f"MLflow UI: mlflow ui --backend-store-uri {core.TRACKING_URI} --port 5000")


if __name__ == "__main__":
    main()
