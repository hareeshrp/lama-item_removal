import os
import cv2
import subprocess

# === USER SETTINGS ===
INPUT_IMAGE = "input_images/input5.png"
INPUT_MASK = "input_images/input5_mask.png"
MODEL_PATH = os.path.abspath("big-lama")
TEMP_DIR = os.path.abspath("temp_run")
OUT_DIR = os.path.abspath("output")
FINAL_OUTPUT = os.path.join(OUT_DIR, "restored_output.png")

# === PREP STEPS ===
def preprocess_and_save(img_path, mask_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"[ERROR] Could not read image: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"[ERROR] Could not read mask: {mask_path}")

    original_shape = img.shape[:2]  # (height, width)

    # Keep aspect ratio and resize to closest multiple of 8 (LaMa requirement)
    h, w = original_shape
    new_h = h - (h % 8)
    new_w = w - (w % 8)
    resize_dims = (new_w, new_h)

    img_resized = cv2.resize(img, resize_dims, interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask, resize_dims, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(save_dir, "image.png"), img_resized)
    cv2.imwrite(os.path.join(save_dir, "image_mask.png"), mask_resized)

    print(f"[✔] Preprocessed image & mask to: {resize_dims}")
    return original_shape  # return original for final restoration

# === RESTORE TO ORIGINAL SIZE ===
def restore_original_size(predicted_path, original_shape, output_path):
    if not os.path.exists(predicted_path):
        raise FileNotFoundError(f"[ERROR] Prediction file not found: {predicted_path}")

    pred = cv2.imread(predicted_path)
    if pred is None:
        raise ValueError(f"[ERROR] Could not read prediction image: {predicted_path}")

    restored = cv2.resize(pred, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, restored)
    print(f"[✔] Restored image saved to: {output_path}")

# === MAIN PIPELINE ===
def run_lama_pipeline():
    print("[1] Preprocessing...")
    original_shape = preprocess_and_save(INPUT_IMAGE, INPUT_MASK, TEMP_DIR)

    print("[2] Running LaMa model...")
    predict_cmd = [
        "python3", "bin/predict.py",
        f"model.path={MODEL_PATH}",
        f"indir={TEMP_DIR}",
        f"outdir={OUT_DIR}"
    ]
    subprocess.run(predict_cmd, check=True)

    print("[3] Post-processing...")
    predicted = os.path.join(OUT_DIR, "image_mask.png")
    restore_original_size(predicted, original_shape, FINAL_OUTPUT)

    print("[✅] LaMa pipeline complete.")

# === RUN ===
if __name__ == "__main__":
    run_lama_pipeline()
