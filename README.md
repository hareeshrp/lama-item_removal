````markdown
Object Removal Pipeline using SAM + LaMa

This project integrates **Segment Anything Model (SAM)** by Meta and **LaMa (Look-at-the-mask)** by SAIC-AI to perform high-quality object removal in images. The goal is to let users semantically select and erase unwanted elements from images with visually realistic results.

---

🔄 Pipeline Summary:

1. **SAM** segments the input image into object masks.
2. User selects relevant masks and combines them into a **single binary mask**.
3. **LaMa** takes the input image and the binary mask to inpaint and remove unwanted content.

---

1. SAM: Segment Anything Model

We use the SAM model to segment an input image into ~150 binary masks, each representing an individual object or region.


🖼️ **Insert Images:**

* `outputs/input4.png`: original image
* `outputs/0.PNG`: example mask
* Grid or overlay of 150 generated masks (optional)

---

2. Combining Selected Masks

The user can visually inspect generated masks and combine a subset into a final binary mask used for inpainting.

```python
# combine_selected_masks.py
selected_files = ["0.png", "2.png", "3.png"]
```

📄 Output: `input4_mask.png` — a binary mask with white areas indicating the regions to remove.

🖼️ **Insert Images:**

* Combined binary mask: `input4_mask.png`

---
3. LaMa Inpainting

A full pipeline script handles:

* Resizing input and mask
* Predicting inpainted output
* Restoring image to original resolution

```bash
python3 lama_pipeline.py
```

📄 Output: `restored_output.png`

🖼️ **Insert Comparison Table:**

| Original        | Combined Mask        | Final Output             |
| --------------- | -------------------- | ------------------------ |
| ![](input4.png) | ![](input4_mask.png) | ![](restored_output.png) |

---


📦 Setup Instructions

<details>
<summary>Environment</summary>

* Python 3.8+
* OpenCV (`cv2`)
* NumPy
* PyTorch 1.10+
* `segment-anything` (Meta AI repo)
* `LaMa` (patched for custom input/output usage)

</details>

🏁 Install SAM

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

🚧 Install LaMa

```bash
git clone https://github.com/advimman/lama.git
cd lama
# Setup a virtual environment and install requirements
conda create -n lama python=3.10
conda activate lama
pip install -r requirements.txt
```

---

💡 Use Cases

* 🏠 Real Estate: Remove furniture or clutter
* 🛒 E-commerce: Clean backgrounds
* 🧪 Research: Create datasets with masked content
* 🎨 Creative: Artistic or visual editing of photos

---

📈 Improvements Coming

* [ ] UI interface (Gradio or Streamlit)
* [ ] Automatic object grouping
* [ ] Batch mode for folders
* [ ] Docker setup
