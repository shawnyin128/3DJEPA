### CV-JEPA-3DGS Project Guide

This project leverages the scene semantic priors learned by **I-JEPA (Image Joint-Embedding Predictive Architecture)** to assist **3D Gaussian Splatting (3DGS)** in achieving high-quality single-view 3D reconstruction.

---

### Core Scripts (script/)

The primary logic is located in the `script/` directory, divided into the following stages:

#### 1. Stage 1: Prior Pre-training
*   **`train_jepa.py`**
    *   **Function**: Trains the JEPA world model on the full dataset.
    *   **Logic**: Learns the geometric patterns of objects in 3D space by predicting feature representations under different camera viewpoints.
    *   **Output**: Saves the trained feature extractor weights (used for Stage 2 initialization).

#### 2. Stage 2: 3DGS Optimization & Validation (Mode B)
This stage utilizes the Stage 1 priors for single-view 3D reconstruction experiments.

*   **`train_gs.py`**
    *   **Function**: **Prior-guided single-view 3D reconstruction**.
    *   **Core Logic**:
        1.  Takes a single ground truth image as input.
        2.  Invokes the JEPA Encoder combined with an action sequence to generate multi-view feature representations.
        3.  Generates a "Saliency Map" based on feature energy and initializes the 3D Gaussian point positions accordingly.
        4.  Performs photometric consistency optimization using only the single training view (View 0).
    *   **Validation**: Renders unseen "future viewpoints" after training to verify geometric generalization.

*   **`train_gs_random.py`**
    *   **Function**: **Experimental Control Group**.
    *   **Logic**: Follows the exact same training pipeline as `train_gs.py`, but uses pure random point initialization. This is used to demonstrate the superiority of the JEPA prior in geometric localization.

---

### Project Structure

*   **`model/`**: Contains core definitions for the JEPA main model, Encoder, Decoder, and Predictor.
*   **`utils/`**:
    *   `gs_utils.py`: 3DGS rendering logic and camera geometric transformation tools.
    *   `dataset_utils.py`: ShapeNet dataset loader.
    *   `action_utils.py`: Tools for encoding camera actions (Yaw/Pitch) and generating sequences.
*   **`vis/`**: Stores visualization comparison plots (GT vs. Pred) generated during training.

---

### Quick Start
1.  Run `python script/train_jepa.py` to complete the first stage of training.
2.  Run `python script/train_gs.py` to observe the 3D reconstruction results guided by the prior.
3.  Compare with results from `python script/train_gs_random.py` to analyze the effectiveness of the prior.

---

### LLM Statement
*This README was generated with the assistance of an AI Language Model (LLM) to ensure clarity, structured documentation, and professional formatting of the project's technical workflow.*
