"""Shared pieces used by train.py and app.py.

Everything that both training and the web app must agree on lives here:
fixed paths, the Bangla token inventory, the conversion between Unicode
words and the left-to-right order in which letters are written, the image
normalisation, and CTC decoding (free and dictionary-guided).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.keras"
MODEL_CARD_PATH = BASE_DIR / "models" / "model_card.json"
LABELS_PATH = BASE_DIR / "labels.json"
LEXICON_PATH = BASE_DIR / "lexicon.txt"
MLFLOW_DIR = BASE_DIR / "artifacts" / "mlflow"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = MLFLOW_DIR / "runs"
EXPERIMENT_NAME = "bangla-word-ocr"

# Every word image is normalised to this size before it reaches the model.
IMG_H, IMG_W = 64, 256

# ---------------------------------------------------------------- script
NUKTA = "়"

# BanglaLekha-Isolated stores each character in a numbered folder.
# Folders 1-50 are the basic characters in the order of the collection form:
# 11 vowels, then 39 consonants (including the nukta letters and the signs
# khanda ta, anusvara, visarga and chandrabindu).
BASIC_CHARS = [
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ",
    "ড" + NUKTA, "ঢ" + NUKTA, "য" + NUKTA, "ৎ", "ং", "ঃ", "ঁ",
]
assert len(BASIC_CHARS) == 50

FOLDER_TO_CHAR = {str(i + 1): ch for i, ch in enumerate(BASIC_CHARS)}

VOWELS = BASIC_CHARS[:11]
CONSONANTS = BASIC_CHARS[11:46]          # ক ... য়
WORD_FINAL_SIGNS = ["ৎ", "ং", "ঃ"]
# Chandrabindu sits on top of another letter, so it is not used inside words.
WORD_BASES = BASIC_CHARS[:49]

# Vowel signs (kar). They are not in BanglaLekha-Isolated; train.py draws them.
PRE_KARS = ["ি", "ে", "ৈ"]               # written to the LEFT of the consonant
POST_KARS = ["া", "ী"]                    # written to the RIGHT, own column
BELOW_KARS = ["ু", "ূ", "ৃ"]              # written UNDER the consonant
KARS = ["া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ"]
O_KAR = "ো"                               # = ে before + া after the consonant

# CTC classes: index 0 is the blank, tokens start at index 1.
TOKENS = WORD_BASES + KARS


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def split_clusters(word: str) -> list[tuple[str, str | None]] | None:
    """Split a word into (letter, vowel sign) pairs.

    Returns None when the word uses something the model cannot draw
    (conjuncts with hasanta, chandrabindu, au-kar, Latin letters ...).
    """
    word = nfc(word.strip())
    base_set = set(WORD_BASES)
    sign_set = set(KARS) | {O_KAR}
    clusters: list[tuple[str, str | None]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if i + 1 < len(word) and word[i + 1] == NUKTA:
            ch += NUKTA
            i += 1
        i += 1
        if ch not in base_set:
            return None
        sign = None
        if i < len(word) and word[i] in sign_set:
            sign = word[i]
            i += 1
        clusters.append((ch, sign))
    return clusters or None


def clusters_to_visual(clusters: list[tuple[str, str | None]]) -> list[str]:
    """Unicode order -> the order a hand writes the marks, left to right."""
    out: list[str] = []
    for base, sign in clusters:
        if sign is None:
            out.append(base)
        elif sign in PRE_KARS:
            out += [sign, base]
        elif sign == O_KAR:
            out += ["ে", base, "া"]
        else:  # া ী ু ূ ৃ
            out += [base, sign]
    return out


def word_to_visual(word: str) -> list[str] | None:
    clusters = split_clusters(word)
    return None if clusters is None else clusters_to_visual(clusters)


def visual_to_word(tokens: list[str]) -> str:
    """Inverse of clusters_to_visual: move left-hand signs back after their letter."""
    out: list[str] = []
    pending: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in PRE_KARS:
            if pending:
                out.append(pending)
            pending = tok
        elif tok in KARS:
            out.append(tok)
        else:
            out.append(tok)
            if pending:
                if pending == "ে" and i + 1 < len(tokens) and tokens[i + 1] == "া":
                    out.append(O_KAR)
                    i += 1
                else:
                    out.append(pending)
                pending = None
        i += 1
    if pending:
        out.append(pending)
    return nfc("".join(out))


# ---------------------------------------------------------------- files
def write_labels(path: Path = LABELS_PATH) -> dict:
    data = {
        "blank_index": 0,
        "tokens": TOKENS,
        "image_height": IMG_H,
        "image_width": IMG_W,
        "banglalekha_folder_to_char": FOLDER_TO_CHAR,
        "note": "class id k (k >= 1) is tokens[k-1]; tokens are in written (visual) order",
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def read_labels(path: Path = LABELS_PATH) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["tokens"]


def read_lexicon(path: Path = LEXICON_PATH) -> list[str]:
    """Words from lexicon.txt that the model can represent, de-duplicated."""
    words: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        for word in line.split():
            word = nfc(word)
            if word not in seen and word_to_visual(word) is not None:
                seen.add(word)
                words.append(word)
    return words


# ---------------------------------------------------------------- images
def white_on_black(gray: np.ndarray) -> np.ndarray:
    """Make sure ink is bright and paper is dark (BanglaLekha and the canvas agree then)."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return 255 - gray if border.mean() > 127 else gray


def crop_ink(gray: np.ndarray, threshold: int = 60, pad: int = 2) -> np.ndarray | None:
    ys, xs = np.nonzero(gray > threshold)
    if len(ys) == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gray.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gray.shape[1])
    return gray[y0:y1, x0:x1]


def fit_word(gray: np.ndarray) -> np.ndarray | None:
    """Crop the ink and place it, aspect kept, on a black IMG_H x IMG_W strip."""
    crop = crop_ink(gray)
    if crop is None or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    h, w = crop.shape
    scale = min((IMG_H - 8) / h, (IMG_W - 8) / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS)
    strip = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    y0 = (IMG_H - nh) // 2
    strip[y0:y0 + nh, 4:4 + nw] = np.asarray(small)
    return strip


def canvas_to_gray(rgba: np.ndarray) -> np.ndarray:
    """Canvas RGBA (white pen on black) -> single channel 0-255."""
    rgb = rgba[..., :3].astype(np.float32).max(axis=-1)
    alpha = rgba[..., 3].astype(np.float32) / 255.0 if rgba.shape[-1] == 4 else 1.0
    return np.clip(rgb * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- decoding
def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))


def greedy_decode(logp: np.ndarray, tokens: list[str]) -> list[str]:
    """Best path: take the top class per time step, merge repeats, drop blanks."""
    best = logp.argmax(axis=-1)
    out, prev = [], -1
    for k in best:
        if k != prev and k != 0:
            out.append(tokens[k - 1])
        prev = k
    return out


def ctc_log_likelihood(logp: np.ndarray, label_seqs: list[list[int]]) -> np.ndarray:
    """log P(label | image) for many candidate labels at once (CTC forward pass)."""
    n = len(label_seqs)
    lengths = np.array([len(s) for s in label_seqs])
    s_max = 2 * lengths.max() + 1
    ext = np.zeros((n, s_max), dtype=np.int64)            # blank, l1, blank, l2, ...
    for i, seq in enumerate(label_seqs):
        ext[i, 1:2 * len(seq):2] = seq
    valid = np.arange(s_max)[None, :] < (2 * lengths + 1)[:, None]
    can_skip = np.zeros((n, s_max), dtype=bool)
    can_skip[:, 2:] = (ext[:, 2:] != 0) & (ext[:, 2:] != ext[:, :-2])

    neg = -1e30
    alpha = np.full((n, s_max), neg)
    alpha[:, 0] = logp[0, 0]
    alpha[:, 1] = np.where(lengths > 0, logp[0, ext[:, 1]], neg)
    for t in range(1, logp.shape[0]):
        stay = alpha
        step = np.concatenate([np.full((n, 1), neg), alpha[:, :-1]], axis=1)
        skip = np.concatenate([np.full((n, 2), neg), alpha[:, :-2]], axis=1)
        skip = np.where(can_skip, skip, neg)
        alpha = np.logaddexp(np.logaddexp(stay, step), skip) + logp[t, ext]
        alpha = np.where(valid, alpha, neg)
    last = 2 * lengths
    rows = np.arange(n)
    return np.logaddexp(alpha[rows, last], np.where(lengths > 0, alpha[rows, last - 1], neg))


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class WordDecoder:
    """Turns the model's per-column scores into a whole Bangla word."""

    def __init__(self, tokens: list[str], lexicon: list[str]):
        self.tokens = tokens
        self.index = {t: i + 1 for i, t in enumerate(tokens)}
        self.set_lexicon(lexicon)

    def set_lexicon(self, words: list[str]) -> None:
        self.words: list[str] = []
        self.visual: list[list[str]] = []
        self.ids: list[list[int]] = []
        for w in dict.fromkeys(nfc(w) for w in words):
            vis = word_to_visual(w)
            if vis and all(t in self.index for t in vis):
                self.words.append(w)
                self.visual.append(vis)
                self.ids.append([self.index[t] for t in vis])

    def decode(self, logits: np.ndarray, use_lexicon: bool = True, top_k: int = 5) -> dict:
        logp = log_softmax(logits.astype(np.float64))
        raw_tokens = greedy_decode(logp, self.tokens)
        raw_word = visual_to_word(raw_tokens)
        result = {"word": raw_word, "raw_word": raw_word, "source": "free decoding",
                  "candidates": []}
        if not use_lexicon or not self.words:
            return result

        scores = ctc_log_likelihood(logp, self.ids)
        order = np.argsort(-scores)[:top_k]
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        result["candidates"] = [(self.words[i], float(probs[i])) for i in order]

        best = int(order[0])
        if raw_word in self.words:
            result["source"] = "dictionary (exact match)"
        else:
            # Snap to the dictionary only when the raw reading is reasonably close.
            dist = edit_distance(raw_tokens, self.visual[best])
            if dist <= max(1, len(self.visual[best]) // 2):
                result["word"] = self.words[best]
                result["source"] = f"dictionary (corrected {dist} mark(s))"
        return result
