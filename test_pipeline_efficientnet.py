import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ─────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"

# ─────────────────────────────────────────
# Configuration  – edit paths as needed
# ─────────────────────────────────────────
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

BODY_PART_MODEL_PATH = os.path.join(THIS_FOLDER, "weights", "EfficientNet_BodyParts_unfreeze_aug.keras")
FRACTURE_MODEL_PATHS = {
    "Elbow":    os.path.join(THIS_FOLDER, "weights", "EfficientNetB3_Elbow_frac_unfreeze_aug.keras"),
    "Hand":     os.path.join(THIS_FOLDER, "weights", "EfficientNetB3_Hand_frac_unfreeze_aug.keras"),
    "Shoulder": os.path.join(THIS_FOLDER, "weights", "EfficientNetB3_Shoulder_frac_unfreeze_aug.keras"),
}

# Body-part label order must match what LabelEncoder produced during training.
# LabelEncoder sorts alphabetically, so:
BODY_PART_CLASSES = ["Elbow", "Hand", "Shoulder"]

IMG_SIZE = 300

# ─────────────────────────────────────────
# Dataset loader  (mirrors training logic)
# ─────────────────────────────────────────
def load_test_dataset(test_dir: str) -> pd.DataFrame:
    """
    Expected directory structure (same as training):
        test_dir/
          <train|test|val>/          ← top-level split folder (iterated but ignored)
            <BodyPart>/              ← e.g. Hand, Elbow, Shoulder
              <patient_id>/
                <anything>_positive/ ← fractured
                <anything>_negative/ ← normal
                  image.jpg
    """
    records = []
    for split_folder in os.listdir(test_dir):
        split_path = os.path.join(test_dir, split_folder)
        if not os.path.isdir(split_path):
            continue
        for body in os.listdir(split_path):
            body_path = os.path.join(split_path, body)
            if not os.path.isdir(body_path):
                continue
            for patient_id in os.listdir(body_path):
                patient_path = os.path.join(body_path, patient_id)
                if not os.path.isdir(patient_path):
                    continue
                for lab in os.listdir(patient_path):
                    suffix = lab.split("_")[-1]
                    if suffix == "positive":
                        fracture_label = "fractured"
                    elif suffix == "negative":
                        fracture_label = "normal"
                    else:
                        continue
                    lab_path = os.path.join(patient_path, lab)
                    for img_file in os.listdir(lab_path):
                        records.append({
                            "image_name":    img_file,
                            "true_part":     body,
                            "true_fracture": fracture_label,
                            "image_path":    os.path.join(lab_path, img_file),
                        })
    return pd.DataFrame(records)

# ─────────────────────────────────────────
# Preprocessing  (mirrors training)
# ─────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image.numpy()

# ─────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────
def load_models():
    print(cyan("\n[*] Loading body-part identification model …"))
    body_model = tf.keras.models.load_model(BODY_PART_MODEL_PATH)
    print(green("    ✓ Body-part model loaded"))

    fracture_models = {}
    for part, path in FRACTURE_MODEL_PATHS.items():
        print(cyan(f"[*] Loading fracture model for {part} …"))
        fracture_models[part] = tf.keras.models.load_model(path)
        print(green(f"    ✓ {part} fracture model loaded"))

    return body_model, fracture_models

# ─────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────
def predict_body_part(body_model, image: np.ndarray) -> str:
    inp = np.expand_dims(image, axis=0)
    probs = body_model.predict(inp, verbose=0)[0]
    idx = np.argmax(probs)
    return BODY_PART_CLASSES[idx]

def predict_fracture(fracture_model, image: np.ndarray) -> str:
    inp = np.expand_dims(image, axis=0)
    prob = fracture_model.predict(inp, verbose=0)[0][0]
    return "fractured" if prob >= 0.5 else "normal"

# ─────────────────────────────────────────
# Formatted table output
# ─────────────────────────────────────────
COL_WIDTHS = [28, 12, 16, 12, 18]   # Name | True Part | Pred Part | True Frac | Pred Frac

def _pad(s, w):
    """Pad / truncate a plain string to width w (stripping ANSI for measurement)."""
    # Visible length (strip ANSI codes for width calculation)
    import re
    visible = re.sub(r'\033\[[0-9;]*m', '', s)
    pad = w - len(visible)
    return s + " " * max(pad, 0)

def print_header():
    headers = ["Name", "Part", "Predicted Part", "Status", "Predicted Status"]
    row = "  ".join(_pad(bold(h), COL_WIDTHS[i]) for i, h in enumerate(headers))
    print(row)
    print("─" * (sum(COL_WIDTHS) + 2 * len(COL_WIDTHS)))

def print_row(name, true_part, pred_part, true_frac, pred_frac):
    part_ok    = true_part == pred_part
    frac_ok    = true_frac == pred_frac
    all_ok     = part_ok and frac_ok
    colour     = green if all_ok else red

    cells = [
        colour(_pad(name,      COL_WIDTHS[0])),
        colour(_pad(true_part, COL_WIDTHS[1])),
        colour(_pad(pred_part, COL_WIDTHS[2])),
        colour(_pad(true_frac, COL_WIDTHS[3])),
        colour(_pad(pred_frac, COL_WIDTHS[4])),
    ]
    print("  ".join(cells))

# ─────────────────────────────────────────
# Metrics printer
# ─────────────────────────────────────────
def print_metrics_section(title: str, y_true, y_pred, y_scores=None):
    print(f"\n{bold(cyan('═' * 60))}")
    print(bold(cyan(f"  {title}")))
    print(bold(cyan('═' * 60)))

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

    print(f"  Accuracy  : {green(f'{acc*100:.2f}%')}")
    print(f"  Precision : {green(f'{prec*100:.2f}%')}")
    print(f"  Recall    : {green(f'{rec*100:.2f}%')}")
    print(f"  F1 Score  : {green(f'{f1*100:.2f}%')}")

    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"  AUC       : {green(f'{auc:.4f}')}")
        except ValueError:
            pass

    print(f"\n{bold('Classification Report:')}")
    print(classification_report(y_true, y_pred, zero_division=0))

def print_confusion_matrix(title, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(bold(f"\n  Confusion Matrix – {title}:"))
    header = "         " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {labels[i]:>6}  " + "  ".join(f"{v:>10}" for v in row)
        print(row_str)

# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────
def run_pipeline(test_dir: str):
    # 1. Load dataset
    print(cyan(f"\n[*] Loading dataset from: {test_dir}"))
    df = load_test_dataset(test_dir)
    if df.empty:
        print(red("[!] No images found. Check directory structure."))
        sys.exit(1)
    print(green(f"    ✓ Found {len(df)} images across {df['true_part'].nunique()} body-part class(es)"))

    # 2. Load models
    body_model, fracture_models = load_models()

    # 3. Run inference
    print(cyan(f"\n[*] Running inference …\n"))
    print_header()

    results = []
    for _, row in df.iterrows():
        try:
            image = preprocess_image(row["image_path"])
        except Exception as e:
            print(red(f"  [!] Could not read {row['image_name']}: {e}"))
            continue

        pred_part    = predict_body_part(body_model, image)
        frac_model   = fracture_models.get(pred_part)
        if frac_model is None:
            pred_frac = "unknown"
        else:
            pred_frac = predict_fracture(frac_model, image)

        print_row(
            row["image_name"],
            row["true_part"],
            pred_part,
            row["true_fracture"],
            pred_frac,
        )

        results.append({
            "image_name":    row["image_name"],
            "true_part":     row["true_part"],
            "pred_part":     pred_part,
            "true_fracture": row["true_fracture"],
            "pred_fracture": pred_frac,
        })

    res_df = pd.DataFrame(results)

    # ── 4. Metrics ──────────────────────────────────────────────────
    # 4a. Body-part identification
    print_metrics_section(
        "BODY PART IDENTIFICATION METRICS",
        res_df["true_part"].values,
        res_df["pred_part"].values,
    )
    print_confusion_matrix(
        "Body Part",
        res_df["true_part"].values,
        res_df["pred_part"].values,
        BODY_PART_CLASSES,
    )

    # 4b. Fracture detection (overall, on images where part was correctly identified)
    valid = res_df[res_df["pred_fracture"] != "unknown"]
    frac_true_bin  = (valid["true_fracture"]  == "fractured").astype(int).values
    frac_pred_bin  = (valid["pred_fracture"]  == "fractured").astype(int).values

    print_metrics_section(
        "FRACTURE DETECTION METRICS  (all images)",
        frac_true_bin,
        frac_pred_bin,
    )
    print_confusion_matrix(
        "Fracture Detection",
        valid["true_fracture"].values,
        valid["pred_fracture"].values,
        ["normal", "fractured"],
    )

    # 4c. Per-part fracture metrics
    for part in BODY_PART_CLASSES:
        part_df = valid[valid["true_part"] == part]
        if part_df.empty:
            continue
        bt = (part_df["true_fracture"] == "fractured").astype(int).values
        bp = (part_df["pred_fracture"] == "fractured").astype(int).values
        print_metrics_section(
            f"FRACTURE DETECTION METRICS  – {part.upper()}",
            bt, bp,
        )

    # 4d. End-to-end accuracy (both part AND fracture correct)
    e2e_correct = (
        (res_df["true_part"]     == res_df["pred_part"]) &
        (res_df["true_fracture"] == res_df["pred_fracture"])
    ).sum()
    print(f"\n{bold(cyan('═' * 60))}")
    print(bold(cyan("  END-TO-END PIPELINE ACCURACY")))
    print(bold(cyan('═' * 60)))
    pct = e2e_correct / len(res_df) * 100
    colour = green if pct >= 80 else yellow if pct >= 60 else red
    print(f"  Correct (part + fracture both right) : {colour(f'{e2e_correct}/{len(res_df)}  ({pct:.2f}%)')}\n")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Fall back to a default path if none is given
        default_dir = os.path.join(THIS_FOLDER, "Dataset-Train")
        print(yellow(f"[!] No test directory supplied. Using default: {default_dir}"))
        test_directory = default_dir
    else:
        test_directory = sys.argv[1]

    if not os.path.isdir(test_directory):
        print(red(f"[!] Directory not found: {test_directory}"))
        sys.exit(1)

    run_pipeline(test_directory)