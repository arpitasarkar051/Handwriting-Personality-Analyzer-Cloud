"""
UI-safe backend adapter for Handwriting Personality Analyzer.

The original project is preserved unchanged in:
    Handwriting_Personality_Analyzer_ML.py

This module contains the same computational pipeline, thresholds, model
settings, feature definitions, descriptions, and clustering configuration.
Only notebook/console I/O has been separated so a cloud UI can call the
pipeline with an uploaded image.
"""
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image. Check the file format (jpg/png work best).")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

def get_text_lines(binary):
    horizontal_proj = np.sum(binary, axis=1)
    threshold = horizontal_proj.max() * 0.08 if horizontal_proj.max() > 0 else 0
    text_rows = horizontal_proj > threshold

    lines, in_line, start = [], False, 0
    for i, val in enumerate(text_rows):
        if val and not in_line:
            start, in_line = i, True
        elif not val and in_line:
            lines.append((start, i))
            in_line = False
    if in_line:
        lines.append((start, len(text_rows)))
    lines = [(s, e) for s, e in lines if e - s > 3]
    return lines

def get_line_spacing_ratio(binary, lines):
    if len(lines) < 2:
        return None
    gaps = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    avg_gap = np.mean(gaps)
    avg_height = np.mean([e - s for s, e in lines])
    return avg_gap / avg_height if avg_height > 0 else None

def get_letter_spacing(binary, lines):
    if not lines:
        return None
    densest = max(lines, key=lambda l: np.sum(binary[l[0]:l[1], :]))
    s, e = densest
    line_img = binary[s:e, :]
    vertical_proj = np.sum(line_img, axis=0)
    threshold = vertical_proj.max() * 0.05 if vertical_proj.max() > 0 else 0
    text_cols = vertical_proj > threshold

    gaps, in_gap, gap_start = [], False, 0
    started = False
    for i, val in enumerate(text_cols):
        if val:
            started = True
        if not val and started and not in_gap:
            gap_start, in_gap = i, True
        elif val and in_gap:
            gaps.append(i - gap_start)
            in_gap = False
    gaps = [g for g in gaps if g > 1]
    return float(np.mean(gaps)) if gaps else 0.0

def get_slant_angle(binary):
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    angles = []
    for c in contours:
        if cv2.contourArea(c) < 15 or len(c) < 5:
            continue
        ellipse = cv2.fitEllipse(c)
        angle = ellipse[2] - 90
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))

def get_pressure_proxy(gray, binary):
    ink_pixels = gray[binary > 0]
    if len(ink_pixels) == 0:
        return 0.0, 0.0
    avg_darkness = 255 - float(np.mean(ink_pixels))
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_width = float(np.mean(dist[binary > 0])) * 2
    return avg_darkness, stroke_width

def extract_features(img, gray, binary):
    lines = get_text_lines(binary)
    line_spacing = get_line_spacing_ratio(binary, lines)
    letter_spacing = get_letter_spacing(binary, lines)
    slant = get_slant_angle(binary)
    darkness, stroke_width = get_pressure_proxy(gray, binary)

    return {
        "lines_detected": len(lines),
        "line_spacing_ratio": line_spacing if line_spacing is not None else 0.5,
        "letter_spacing_px": letter_spacing if letter_spacing is not None else 8.0,
        "slant_angle_deg": slant,
        "ink_darkness": darkness,
        "stroke_width_px": stroke_width,
    }, lines

def weak_label_slant(slant):
    if slant > 8: return "Right-leaning"
    if slant < -8: return "Left-leaning"
    return "Upright"

def weak_label_letter_spacing(ls):
    if ls > 12: return "Wide"
    if ls < 5: return "Tight"
    return "Moderate"

def weak_label_line_spacing(lsr):
    if lsr > 0.8: return "Generous"
    if lsr < 0.3: return "Cramped"
    return "Even"

def weak_label_pressure(score):
    if score > 90: return "Heavy"
    if score < 40: return "Light"
    return "Medium"

def generate_synthetic_dataset(n_samples=4000, label_noise=0.12, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    slant = rng.uniform(-25, 25, n_samples)
    letter_spacing = rng.uniform(0, 22, n_samples)
    line_spacing_ratio = rng.uniform(0, 1.3, n_samples)
    ink_darkness = rng.uniform(30, 220, n_samples)
    stroke_width = rng.uniform(0.5, 14, n_samples)
    pressure_score = ink_darkness * 0.5 + stroke_width * 5

    X = np.column_stack([
        slant, letter_spacing, line_spacing_ratio, ink_darkness, stroke_width
    ])

    y_slant = np.array([weak_label_slant(v) for v in slant])
    y_letter = np.array([weak_label_letter_spacing(v) for v in letter_spacing])
    y_line = np.array([weak_label_line_spacing(v) for v in line_spacing_ratio])
    y_pressure = np.array([weak_label_pressure(v) for v in pressure_score])

    targets = {
        "Slant": y_slant,
        "Letter spacing": y_letter,
        "Line spacing": y_line,
        "Pressure": y_pressure,
    }

    for name, y in targets.items():
        classes = np.unique(y)
        flip_mask = rng.random(n_samples) < label_noise
        random_labels = rng.choice(classes, size=n_samples)
        y[flip_mask] = random_labels[flip_mask]

    feature_names = [
        "slant_angle_deg", "letter_spacing_px", "line_spacing_ratio",
        "ink_darkness", "stroke_width_px"
    ]
    return X, targets, feature_names

def train_pipeline():
    X, y_dict, feature_names = generate_synthetic_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    trained_models = {}
    results_table = []

    for trait_name, y in y_dict.items():
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=RANDOM_SEED,
            stratify=y_enc
        )

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_SEED
        )
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_test))

        if rf_acc >= lr_acc:
            best_model, best_name, best_acc = rf, "Random Forest", rf_acc
        else:
            best_model, best_name, best_acc = lr, "Logistic Regression", lr_acc

        trained_models[trait_name] = {
            "model": best_model,
            "encoder": encoder,
            "algo": best_name,
        }
        results_table.append((trait_name, rf_acc, lr_acc, best_name))

    return {
        "X": X,
        "y_dict": y_dict,
        "feature_names": feature_names,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "trained_models": trained_models,
        "results_table": results_table,
    }

def prepare_cluster_model(X_scaled):
    N_CLUSTERS = 4
    kmeans = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    return kmeans, cluster_labels, pca, X_pca

TRAIT_DESCRIPTIONS = {
    ("Slant", "Right-leaning"): "Outgoing and expressive — you tend to act on impulse and enjoy connecting with people.",
    ("Slant", "Left-leaning"): "Reserved and introspective — you prefer thinking things through before acting.",
    ("Slant", "Upright"): "Balanced and practical — you value logic and keep emotions in check.",
    ("Letter spacing", "Wide"): "Values independence and personal space; thinks before speaking.",
    ("Letter spacing", "Tight"): "Sociable and comfortable with closeness; may act impulsively in groups.",
    ("Letter spacing", "Moderate"): "Comfortable in most social settings, adapts easily.",
    ("Line spacing", "Generous"): "Clear, organized thinker who likes structure and planning.",
    ("Line spacing", "Cramped"): "A multitasker whose mind juggles many things at once.",
    ("Line spacing", "Even"): "Steady and methodical approach to tasks.",
    ("Pressure", "Heavy"): "Intense and emotionally engaged; takes things seriously.",
    ("Pressure", "Light"): "Sensitive and easygoing; doesn't hold onto stress for long.",
    ("Pressure", "Medium"): "Emotionally balanced with a calm, steady energy.",
}

CLUSTER_BLURBS = {
    0: "archetype cluster leaning toward bold, high-energy strokes",
    1: "archetype cluster leaning toward measured, compact writing",
    2: "archetype cluster leaning toward airy, spaced-out composition",
    3: "archetype cluster leaning toward dense, heavy-pressure writing",
}

def analyze_features(features, pipeline):
    sample_vec = np.array([[
        features["slant_angle_deg"],
        features["letter_spacing_px"],
        features["line_spacing_ratio"],
        features["ink_darkness"],
        features["stroke_width_px"],
    ]])

    sample_scaled = pipeline["scaler"].transform(sample_vec)
    kmeans, cluster_labels, pca, X_pca = prepare_cluster_model(pipeline["X_scaled"])

    sample_cluster = int(kmeans.predict(sample_scaled)[0])
    sample_pca = pca.transform(sample_scaled)

    predicted_traits = {}
    for trait_name, info in pipeline["trained_models"].items():
        pred_class_idx = info["model"].predict(sample_scaled)[0]
        label = info["encoder"].inverse_transform([pred_class_idx])[0]
        description = TRAIT_DESCRIPTIONS.get((trait_name, label), "")
        predicted_traits[trait_name] = (label, description, info["algo"])

    return {
        "sample_vec": sample_vec,
        "sample_scaled": sample_scaled,
        "kmeans": kmeans,
        "cluster_labels": cluster_labels,
        "pca": pca,
        "X_pca": X_pca,
        "sample_cluster": sample_cluster,
        "sample_pca": sample_pca,
        "predicted_traits": predicted_traits,
        "cluster_blurb": CLUSTER_BLURBS.get(sample_cluster, ""),
    }

def normalize(value, lo, hi):
    if value is None:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0, 1))

def get_radar_values(features):
    return [
        normalize(features["slant_angle_deg"], -20, 20),
        normalize(features["letter_spacing_px"], 0, 20),
        normalize(features["line_spacing_ratio"] or 0, 0, 1.2),
        normalize(
            features["ink_darkness"] * 0.5 + features["stroke_width_px"] * 5,
            0, 150
        ),
    ]
