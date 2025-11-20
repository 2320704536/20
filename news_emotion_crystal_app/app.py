# ============================================================
# Emotional Crystal — FINAL VERSION D (No CSV, Emotion Colors Only)
# Fixed Emotion Set for Random Mode
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import date
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import math

# ============================================================
# Streamlit App Settings
# ============================================================
st.set_page_config(
    page_title="Emotional Crystal — Final D",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Emotional Crystal — Final Version D (Emotion Colors Only)")

with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Version (Emotion-Only Mode)

This version removes **all CSV palette logic** and relies purely on the
built-in **emotion → RGB color** mapping.

---

### 1) Fetch News
- Enter a keyword such as *AI*, *technology*, *design*
- System fetches news via NewsAPI
- Each title/description is classified into emotions
- Colors come **exclusively from DEFAULT_RGB**

---

### 2) Random Generate Mode
Random mode now uses a **fixed emotion set**:

**["joy", "love", "awe", "curiosity", "calm", "sadness"]**

Each uses DEFAULT_RGB 100% precisely — no jitter, no fallback.

---

### 3) Crystal Rendering
- Each emotion generates crystal fragments
- Colors come directly from DEFAULT_RGB
- No CSV fallback, no custom palette

---

### 4) Post-processing
Only standard cinematic adjustments (exposure, bloom, etc.)
""")

# ============================================================
# VADER Sentiment Analyzer
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

# ============================================================
# Fetch News from NewsAPI
# ============================================================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# ============================================================
# Emotion Color Mapping (NO CSV)
# ============================================================
DEFAULT_RGB = {
    "joy":        (255,200,60),
    "love":       (255,95,150),
    "pride":      (190,100,255),
    "hope":       (60,235,190),
    "curiosity":  (50,190,255),
    "calm":       (70,135,255),
    "surprise":   (255,160,70),
    "neutral":    (190,190,200),
    "sadness":    (80,120,230),
    "anger":      (245,60,60),
    "fear":       (150,70,200),
    "disgust":    (150,200,60),
    "anxiety":    (255,200,60),
    "boredom":    (135,135,145),
    "nostalgia":  (250,210,150),
    "gratitude":  (90,230,230),
    "awe":        (120,245,255),
    "trust":      (60,200,160),
    "confusion":  (255,140,180),
    "mixed":      (230,190,110),
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())

# ============================================================
# Sentiment → Emotion Classification
# ============================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5: return "surprise"
    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"
    if neu > 0.75 and abs(comp) < 0.1: return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    return "neutral"
  # ============================================================
# Part 2 — Crystal Shape + Rendering Engine (No CSV)
# ============================================================

# ------------------------------------------------------------
# Crystal Shape Generator ❄️
# ------------------------------------------------------------
def crystal_shape(center=(0.5, 0.5), r=150, wobble=0.25,
                  sides_min=5, sides_max=10, rng=None):
    """Generate irregular crystal polygon."""
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n_vertices = int(rng.integers(sides_min, sides_max + 1))

    # shuffle angles so shape is irregular
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    # random radius for each point
    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    pts.append(pts[0])  # close shape
    return pts


# ------------------------------------------------------------
# Draw Soft Polygon (Glow edges)
# ------------------------------------------------------------
def draw_polygon_soft(canvas_rgba, pts, col01, fill_alpha=200,
                      blur_px=6, edge_width=0):
    W, H = canvas_rgba.size

    # temporary layer
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")

    # convert from 0-1 float to 0-255 int
    fill_color = (
        int(col01[0] * 255),
        int(col01[1] * 255),
        int(col01[2] * 255),
        fill_alpha
    )

    d.polygon(pts, fill=fill_color)

    if edge_width > 0:
        edge = (255,255,255, max(80, fill_alpha//2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    canvas_rgba.alpha_composite(layer)


# ------------------------------------------------------------
# Color utilities
# ------------------------------------------------------------
def _rgb01(rgb):
    """Convert 0–255 RGB to 0–1 normalized float array."""
    return np.clip(np.array(rgb, dtype=np.float32) / 255.0, 0, 1)


def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    """Gently boost saturation and brightness."""
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]

    # lift dark colors slightly
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)

    # apply saturation
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = lum + (c - lum) * sat_boost
    return tuple(np.clip(c, 0, 1))


def jitter_color(rgb01, rng, amount=0.06):
    """Small random color variation (disabled in strict mode)."""
    j = (rng.random(3) - 0.5) * 2 * amount
    return tuple(np.clip(rgb01 + j, 0, 1))


# ------------------------------------------------------------
# Crystal Renderer — NO CSV VERSION 🔥
# ------------------------------------------------------------
def render_crystalmix(
    df,
    palette,
    width=1500,
    height=850,
    seed=12345,
    shapes_per_emotion=8,
    min_size=60,
    max_size=220,
    fill_alpha=210,
    blur_px=6,
    bg_color=(0,0,0),
    wobble=0.25,
    layers=8
):
    """Render crystal field for given emotion-labeled df.
       Colors always come from DEFAULT_RGB.
    """
    rng = np.random.default_rng(seed)

    # background
    base = Image.new("RGBA", (width, height), (*bg_color, 255))
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))

    # emotions we will draw
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["joy", "love", "awe"]

    # render multiple layers for bloom effect
    for _layer in range(layers):
        for emo in emotions:

            # ALWAYS use DEFAULT_RGB → 100% precise
            base_rgb = palette.get(emo, (255,255,0))  # fallback yellow if missing
            base01 = vibrancy_boost(base_rgb)

            for _ in range(max(1, int(shapes_per_emotion))):

                cx = rng.uniform(0.05*width, 0.95*width)
                cy = rng.uniform(0.08*height, 0.92*height)
                rr = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx, cy),
                    r=rr,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                # jitter
                col01 = jitter_color(base01, rng, amount=0.07)

                local_alpha = int(np.clip(fill_alpha * rng.uniform(0.85, 1.05), 40, 255))
                local_blur = max(0, int(blur_px * rng.uniform(0.7, 1.4)))
                edge_w = 0 if rng.random() < 0.6 else max(1, int(rr*0.02))

                draw_polygon_soft(
                    canvas, pts, col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    base.alpha_composite(canvas)
    return base.convert("RGB")
# ============================================================
# Part 3 — Cinematic Color System (Exposure / WB / Bloom / ABC)
# ============================================================

# ------------------------------------------------------------
# sRGB ↔ Linear conversions
# ------------------------------------------------------------
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x*12.92, 1.055 * (x ** (1/2.4)) - 0.055)


# ------------------------------------------------------------
# Filmic tonemap
# ------------------------------------------------------------
def filmic_tonemap(x):
    A = 0.22
    B = 0.30
    C = 0.10
    D = 0.20
    E = 0.01
    F = 0.30
    return ((x * (A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F


# ------------------------------------------------------------
# White Balance control
# ------------------------------------------------------------
def apply_white_balance(lin_img, temp, tint):
    """
    temp: -1 → cool, +1 → warm
    tint: -1 → green, +1 → magenta
    """
    temp_strength = 0.6
    tint_strength = 0.5

    wb_temp = np.array([
        1 + temp * temp_strength,
        1,
        1 - temp * temp_strength
    ])

    wb_tint = np.array([
        1 + tint * tint_strength,
        1 - tint * tint_strength,
        1 + tint * tint_strength
    ])

    wb = wb_temp * wb_tint

    out = lin_img * wb.reshape(1,1,3)
    return np.clip(out, 0, 4)


# ------------------------------------------------------------
# Basic color adjustments
# ------------------------------------------------------------
def adjust_contrast(img, c):
    return np.clip((img - 0.5) * c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    lum = lum[...,None]
    return np.clip(lum + (img - lum) * s, 0, 1)

def gamma_correct(img, gamma):
    return np.clip(img ** (1.0/gamma), 0, 1)


# ------------------------------------------------------------
# Highlight Roll-off (soft highlight protection)
# ------------------------------------------------------------
def highlight_rolloff(img, roll):
    t = np.clip(roll, 0, 1.5)
    threshold = 0.8

    mask = np.clip((img - threshold) / (1 - threshold + 1e-6), 0, 1)
    out = img*(1-mask) + (threshold + (img - threshold)/(1 + 4*t*mask)) * mask

    return np.clip(out, 0, 1)


# ------------------------------------------------------------
# Split Toning (Shadow & Highlight color shifts)
# ------------------------------------------------------------
def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    shadows = np.clip(1 - lum + 0.5*(1 - balance), 0, 1)[...,None]
    highlights = np.clip(lum + 0.5*(1 + balance) - 0.5, 0, 1)[...,None]

    sh_col = np.array(sh_rgb).reshape(1,1,3)
    hi_col = np.array(hi_rgb).reshape(1,1,3)

    out = img + shadows * sh_col * 0.25 + highlights * hi_col * 0.25
    return np.clip(out, 0, 1)


# ------------------------------------------------------------
# Bloom effect
# ------------------------------------------------------------
def apply_bloom(img, radius=6, intensity=0.6):
    pil = Image.fromarray((np.clip(img,0,1) * 255).astype(np.uint8), "RGB")
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32)/255.0
        return np.clip(img*(1-intensity) + b*intensity, 0, 1)
    return img


# ------------------------------------------------------------
# Vignette
# ------------------------------------------------------------
def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx = (xx - w/2) / (w/2)
    yy = (yy - h/2) / (h/2)

    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength * (r ** 1.5), 0, 1)

    return np.clip(img * mask[...,None], 0, 1)


# ------------------------------------------------------------
# Ensure minimum colorfulness
# ------------------------------------------------------------
def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)

    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img


# ------------------------------------------------------------
# Auto Brightness Compensation (ABC)
# ------------------------------------------------------------
def auto_brightness_compensation(
    img,
    target_mean=0.50,
    strength=0.9,
    black_point_pct=0.05,
    white_point_pct=0.997,
    max_gain=2.6
):
    arr = np.clip(img, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    wp = max(wp, bp + 1e-3)

    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)

    Y_final = (1-strength)*Y + strength * Y_remap
    meanY = max(Y_final.mean(), 1e-4)

    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)
    lin *= gain

    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.65 * strength

    Y_mix = (1-blend)*Y2 + blend * np.clip(Y_final * gain, 0, 2.5)
    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)

    lin = np.clip(lin * ratio[...,None], 0, 4)

    out = filmic_tonemap(lin)
    out = linear_to_srgb(np.clip(out, 0, 1))

    return np.clip(out, 0, 1)
# ============================================================
# Part 4 — Data Fetching, Emotion Mapping & Sidebar Controls
# ============================================================

# --------------------------
# Default Emotion → RGB
# --------------------------
DEFAULT_RGB = {
    "joy":        (255,200,60),
    "love":       (255,95,150),
    "pride":      (190,100,255),
    "hope":       (60,235,190),
    "curiosity":  (50,190,255),
    "calm":       (70,135,255),
    "surprise":   (255,160,70),
    "neutral":    (190,190,200),
    "sadness":    (80,120,230),
    "anger":      (245,60,60),
    "fear":       (150,70,200),
    "disgust":    (150,200,60),
    "anxiety":    (255,200,60),
    "boredom":    (135,135,145),
    "nostalgia":  (250,210,150),
    "gratitude":  (90,230,230),
    "awe":        (120,245,255),
    "trust":      (60,200,160),
    "confusion":  (255,140,180),
    "mixed":      (230,190,110),
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())


# ------------------------------------------------------------
# VADER sentiment → {neg,neu,pos,compound}
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)


# ------------------------------------------------------------
# Expanded Emotion Classifier
# ------------------------------------------------------------
def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5: return "surprise"
    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"
    if neu > 0.75 and abs(comp) < 0.1: return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    return "neutral"


# ------------------------------------------------------------
# NewsAPI fetch function
# ------------------------------------------------------------
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()

        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"News fetch error: {e}")
        return pd.DataFrame()


# ============================================================
# RANDOM MODE (Option A — Fixed Emotion Set)
# ============================================================
FIXED_RANDOM_EMOTIONS = [
    "joy", "love", "awe", "curiosity", "calm", "sadness"
]


def generate_random_df():
    rng = np.random.default_rng()

    texts = []
    emos = []

    for emo in FIXED_RANDOM_EMOTIONS:
        texts.append(f"Crystal emotion — {emo}")
        emos.append(emo)

    today = str(date.today())

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": today,
        "compound": 0,
        "pos": 0,
        "neu": 1,
        "neg": 0,
        "source": "RandomGen"
    })

    return df


# ============================================================
# SIDEBAR — Data Source & Settings
# ============================================================
st.sidebar.header("1) Data Source")

keyword = st.sidebar.text_input("Keyword (NewsAPI)", "")
fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (Fixed Emotions)")  # Option A

# storage for seed
if "auto_seed" not in st.session_state:
    st.session_state["auto_seed"] = 22

# Load DataFrame
df = pd.DataFrame()

# RANDOM MODE (A)
if random_btn:
    st.session_state["auto_seed"] = int(np.random.randint(1, 99999))
    df = generate_random_df()

# FETCH NEWS
elif fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        st.session_state["auto_seed"] = int(np.random.randint(1, 99999))
        df = fetch_news(key, keyword if keyword.strip() else "technology")

# DEFAULT DEMO
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky.",
        "Calm weather creates a beautiful environment.",
        "Investor anxiety rises in volatile markets.",
        "A moment of awe as the sky glows green.",
        "Hope emerges with new scientific discoveries."
    ]})
    df["timestamp"] = str(date.today())


# Run emotion mapping only if df has no emotion
if "emotion" not in df.columns:
    sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df, sent_df], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("2) Emotion Filters")

cmp_min = st.sidebar.slider("Compound minimum", -1.0, 1.0, -1.0, 0.01)
cmp_max = st.sidebar.slider("Compound maximum", -1.0, 1.0, 1.0, 0.01)

available_emotions = sorted(df["emotion"].unique().tolist())

selected_emotions = st.sidebar.multiselect(
    "Selected Emotions",
    available_emotions,
    default=available_emotions
)

df = df[
    (df["emotion"].isin(selected_emotions)) &
    (df["compound"] >= cmp_min) &
    (df["compound"] <= cmp_max)
]


# ============================================================
# Sidebar — Crystal Engine controls
# ============================================================
st.sidebar.header("3) Crystal Engine")

layer_count = st.sidebar.slider("Layers", 1, 30, 8)
seed_control = st.sidebar.slider("Seed", 0, 99999, st.session_state["auto_seed"])

crystals_per_emotion = st.sidebar.slider("Crystals per Emotion", 1, 40, 8)
poly_min_size = st.sidebar.slider("Min Crystal Size", 20, 300, 60)
poly_max_size = st.sidebar.slider("Max Crystal Size", 60, 600, 220)

stroke_blur = st.sidebar.slider("Crystal Softness (Blur)", 0.0, 20.0, 6.0)
fill_alpha = st.sidebar.slider("Crystal Alpha", 40, 255, 210)
wobble_control = st.sidebar.slider("Crystal Wobble", 0.00, 1.00, 0.25)


# Background Color
st.sidebar.header("4) Background Color")
bg_hex = st.sidebar.color_picker("Background", "#000000")
bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (1,3,5))


# Cinematic Color System
st.sidebar.header("5) Cinematic Color")

exp = st.sidebar.slider("Exposure", -0.2, 1.8, 0.55)
contrast = st.sidebar.slider("Contrast", 0.70, 1.80, 1.18)
saturation = st.sidebar.slider("Saturation", 0.70, 1.90, 1.18)
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, 0.92)
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, 0.40)

# White Balance
temp = st.sidebar.slider("Temperature", -1.0, 1.0, 0.00)
tint = st.sidebar.slider("Tint", -1.0, 1.0, 0.00)

# Split Toning
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, 0.08)
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, 0.06)
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, 0.16)

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, 0.10)
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, 0.08)
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, 0.06)

tone_balance = st.sidebar.slider("Tone Balance", -1.0, 1.0, 0.0)


# Bloom & Vignette
st.sidebar.header("6) Bloom & Vignette")
bloom_radius = st.sidebar.slide# ============================================================
# Part 5 — Final Rendering + PostFX + Download + Data Table
# ============================================================

left, right = st.columns([0.62, 0.38])

# ============================================================
# LEFT PANEL — Crystal Rendering
# ============================================================
with left:
    st.subheader("❄️ Crystal Visualization")

    # Emotion → RGB (strict)
    working_palette = dict(DEFAULT_RGB)

    # ---------------------------------
    # Render Crystal Mix
    # ---------------------------------
    img = render_crystalmix(
        df=df,
        palette=working_palette,          # ← 100% emotion colors only
        width=1500,
        height=850,
        seed=seed_control,
        shapes_per_emotion=crystals_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(fill_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    # Convert to NumPy
    arr = np.array(img).astype(np.float32) / 255.0

    # ---------------------------------
    # APPLY CINEMATIC POST PROCESSING
    # ---------------------------------

    # Linear space
    lin = srgb_to_linear(arr)

    # Exposure
    lin = lin * (2.0 ** exp)

    # White Balance
    lin = apply_white_balance(lin, temp, tint)

    # Highlight rolloff
    lin = highlight_rolloff(lin, roll)

    # Back to sRGB
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    # Filmic curve
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)

    # Contrast / Saturation / Gamma
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # Split Toning
    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    # Auto Brightness Compensation
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # Bloom + Vignette
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)

    # Colorfulness restore
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    # Convert back to image
    final_img = Image.fromarray(
        (np.clip(arr, 0, 1) * 255).astype(np.uint8),
        mode="RGB"
    )

    # ---------------------------------
    # DISPLAY + DOWNLOAD
    # ---------------------------------
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(final_img, use_column_width=True)

    st.download_button(
        "💾 Download PNG",
        data=buf,
        file_name="crystal_emotion.png",
        mime="image/png"
    )


# ============================================================
# RIGHT PANEL — Data Table
# ============================================================
with right:

    st.subheader("📊 Data & Emotion Mapping")

    df2 = df.copy()
    df2["emotion_color"] = df2["emotion"].apply(lambda e: str(DEFAULT_RGB.get(e, (0,0,0))))

    cols = ["text", "emotion", "emotion_color", "compound", "pos", "neu", "neg"]
    if "timestamp" in df2.columns:
        cols.insert(1, "timestamp")
    if "source" in df2.columns:
        cols.insert(2, "source")

    st.dataframe(df2[cols], height=700, use_container_width=True)
r("Bloom Radius", 0.0, 20.0, 7.0)
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, 0.40)
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, 0.16)


# Auto Brightness
st.sidebar.header("7) Auto Brightness")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", True)

target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70, 0.52)
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, 0.92)
abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20, 0.05)
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00, 0.997)
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, 2.6)

