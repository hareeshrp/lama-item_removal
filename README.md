# Object Removal Pipeline using SAM + LaMa

This project integrates **Segment Anything Model (SAM)** by Meta and **LaMa (Look-at-the-mask)** by SAIC-AI to perform high-quality object removal in images. The goal is to let users semantically select and erase unwanted elements from images with visually realistic results.

---

## Pipeline Summary

1. **SAM** segments the input image into object masks.
2. The user selects relevant masks and combines them into a **single binary mask**.
3. **LaMa** takes the input image and the binary mask to inpaint and remove unwanted content.

---

## 1. SAM: Segment Anything Model

We use the SAM model to segment an input image into \~150 binary masks, each representing an individual object or region.

**Example Run:**

```bash
python3 scripts/amg.py \
  --checkpoint path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --input input_images/input4.png
```

**Sample Images:**

* `outputs/input4.png`: original image
* `outputs/0.PNG`: example mask
* Grid or overlay of all generated masks (optional)

---

## 2. Combining Selected Masks

The user visually inspects the generated masks and combines the selected ones into a final binary mask used for inpainting.

**combine\_selected\_masks.py**

```python
selected_files = ["0.png", "2.png", "3.png"]
```

**Output:**

* `input4_mask.png`: A binary mask with white areas indicating the regions to remove.

**Insert Images:**

* Combined mask preview (`input4_mask.png`)

---

## 3. LaMa Inpainting

A full pipeline script (`lama_pipeline.py`) handles:

* Resizing input and mask
* Predicting inpainted output
* Restoring image to original resolution

**Run:**

```bash
python3 lama_pipeline.py
```

**Output:**

* `restored_output.png`: Final inpainted image

---

## Results

| Original        | Combined Mask        | Final Output             |
| --------------- | -------------------- | ------------------------ |
| ![](outputs/input4.png) | ![](outputs/input4_mask.png) | ![](outputs/restored_output.png) |

---

## Setup Instructions

<details>
<summary>Environment Requirements</summary>

* Python 3.8+
* OpenCV (`cv2`)
* NumPy
* PyTorch 1.10+
* `segment-anything` (Meta AI repo)
* `LaMa` (patched for CLI usage)

</details>

### Install SAM

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

### Install LaMa

```bash
git clone https://github.com/advimman/lama.git
cd lama
# Set up Python environment
conda create -n lama python=3.10 -y
conda activate lama
pip install -r requirements.txt
```

---

## Use Cases

* Real Estate: Remove furniture or clutter
* E-commerce: Clean product backgrounds
* Research: Mask-sensitive dataset creation
* Creative Editing: Artistic or visual cleanup

