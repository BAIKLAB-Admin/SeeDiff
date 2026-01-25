text
# SeeDiff: Off-the-Shelf Seeded Mask Generation from Diffusion Models

**SeeDiff** is a training-free model that leverages the internal attention mechanisms of **Stable Diffusion** to simultaneously generate high-quality images and their corresponding precision segmentation masks.  
By aggregating cross-attention and self-attention maps directly from the diffusion process, SeeDiff extracts semantic layouts without any additional supervision.

---

## 🚀 Key Features

- **Training-Free**: Utilizes pre-trained Stable Diffusion v1-4 without additional fine-tuning.  
- **Attention Aggregation**: Combines multi-resolution attention maps (8×8, 16×16, 32×32, 64×64) to capture both global structures and fine-grained details.  
- **Refinement with PAMR**: Includes a *Pixel-Adaptive Mask Refinement (PAMR)* module to ensure mask boundaries align perfectly with generated image structures.  
- **Interactive Demo**: Built-in **Gradio** application for real-time testing and visualization.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BAIKLAB-Admin/SeeDiff.git
cd SeeDiff
```

### 2. Install Dependencies
This project requires Python 3.8+ and the libraries listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## 💻 Usage
### 1. Interactive Web Demo (Gradio)
Launch the interactive interface to generate image–mask pairs via simple text input:

```bash
python src/seediff_app.py
```
The demo will be available at:
👉 http://localhost:8020

Simply enter a class name (e.g., "dog", "cat", "car") to generate image and mask results.

### 2. Dataset Generation (CLI)
Generate large-scale synthetic datasets for training downstream segmentation tasks:

```bash
python src/seediff_main_origin.py \
    --classes "dog" \
    --image_number 100 \
    --gpu_num 0 \
    --thread_num 4 \
    --output "./output" \
    --MY_TOKEN "YOUR_HUGGINGFACE_TOKEN"
```

## 🔬 Project Structure

* `src/seediff_app.py`: Main script for the Gradio-based interactive demo.
* `src/pamr.py`: Implementation of the Pixel-Adaptive Mask Refinement (PAMR) module.
* `src/ptp_utils.py`: Utilities for controlling the diffusion process and managing the attention store.
* `src/seediff_main_origin.py`: Script for bulk dataset generation and mask saving.
* `src/seq_aligner.py`: Token alignment and mapping logic for prompt-based control.

## 📝 Citation & Authors

If you find this project useful in your research, please consider citing our paper:

- **Title**: [SeeDiff: Simultaneous Image and Segmentation Mask Generation](https://arxiv.org/abs/2507.19808)
- **Authors**: Joon Hyun Park¹, Kumju Jo², Sungyong Baik†

- **Paper**: [arXiv:2507.19808](https://arxiv.org/abs/2507.19808)

### 📜 BibTeX
```bibtex
@article{SeeDiff,
  title={SeeDiff: Off-the-Shelf Seeded Mask Generation from Diffusion Models}, 
  author={Joon Hyun Park and Kumju Jo and Sungyong Baik},
  year={2025},
  eprint={2507.19808},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={[https://arxiv.org/abs/2507.19808](https://arxiv.org/abs/2507.19808)}
}