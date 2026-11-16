"""Streamlit page: draw one Bangla word, get the whole word back.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import ocr_core as core

st.set_page_config(page_title="Bangla Word Reader", page_icon="✍️", layout="wide")

EXAMPLES = ["মানুষ", "জল", "বাড়ি", "পাখি", "নদী", "মেয়ে", "বিড়াল", "আকাশ", "বোন", "পৃথিবী"]


@st.cache_resource(show_spinner="Loading the model ...")
def load_model():
    return keras.models.load_model(core.MODEL_PATH, compile=False)


@st.cache_data
def load_lexicon() -> list[str]:
    return core.read_lexicon()


def model_card() -> dict:
    if core.MODEL_CARD_PATH.exists():
        return json.loads(core.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    return {}


if not core.MODEL_PATH.exists():
    st.error(f"No trained model at `{core.MODEL_PATH.relative_to(core.BASE_DIR)}`. Run `python train.py` first.")
    st.stop()

model = load_model()
tokens = core.read_labels()
if model.output_shape[-1] != len(tokens) + 1:
    st.error("labels.json does not match the model. Re-run train.py so both come from the same run.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    pen = st.slider("Pen width", 4, 20, 9)
    use_dictionary = st.toggle("Dictionary-guided decoding", value=True,
                               help="Score every dictionary word against the model output and pick the "
                                    "most likely one when the raw reading is close to it.")
    extra = st.text_area("Extra dictionary words (space separated)", placeholder="যেমন: কাগজ বালিশ")
    show_input = st.checkbox("Show what the model sees", value=True)
    card = model_card()
    if card:
        st.caption(f"Model from run **{card.get('run_name')}** · validation CER "
                   f"{card.get('val_cer', 0):.3f}")

lexicon = load_lexicon() + extra.split()
decoder = core.WordDecoder(tokens, lexicon)

# --------------------------------------------------------------------- page
st.title("বাংলা হাতের লেখা শব্দ চেনা")
st.caption("Bangla handwritten word recognition · CRNN + CTC trained on BanglaLekha-Isolated · "
           "Redowanul Akbar · ID 1000129290")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

left, right = st.columns([3, 2], gap="large")
with left:
    st.write("Write **one word** across the box, as you would on paper (headline on top is fine).")
    canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=pen, stroke_color="#FFFFFF",
                       background_color="#000000", width=760, height=260, drawing_mode="freedraw",
                       key=f"canvas_{st.session_state.canvas_key}")
    b1, b2, _ = st.columns([1, 1, 3])
    read = b1.button("Read word", type="primary", use_container_width=True)
    if b2.button("Clear", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()
    st.caption("Try: " + "  ·  ".join(EXAMPLES))

with right:
    if not read:
        st.info("Write a word on the left, then press **Read word**.")
    elif canvas.image_data is None:
        st.warning("The canvas is not ready yet. Draw something and press the button again.")
    else:
        strip = core.fit_word(core.canvas_to_gray(canvas.image_data))
        if strip is None:
            st.warning("The canvas is empty. Write a word first.")
        else:
            logits = model(strip[None, ..., None].astype("float32"), training=False)
            result = decoder.decode(keras.ops.convert_to_numpy(logits)[0], use_lexicon=use_dictionary)

            st.markdown("Predicted word")
            st.markdown(f"<div style='font-size:72px;line-height:1.1;font-weight:600'>{result['word'] or '—'}</div>",
                        unsafe_allow_html=True)
            st.caption(f"Decided by: {result['source']}")
            st.write(f"Raw model reading (no dictionary): **{result['raw_word'] or '—'}**")

            if result["candidates"]:
                table = pd.DataFrame(result["candidates"], columns=["dictionary word", "probability"])
                table["probability"] = table["probability"].map(lambda p: f"{p:.1%}")
                st.dataframe(table, hide_index=True, use_container_width=True)
            if show_input:
                st.image(strip, caption="Normalised 64 × 256 input", width=512, clamp=True)
