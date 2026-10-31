# Driver Drowsiness Detection

**Subject:** Natural Language Processing (NLP)  
**Project:** Driver Drowsiness Detection  
**Institution:** Vellore Institute of Technology  
**Submitted to:** [Faculty Name]

---

## Research Paper

**Real-Time, Edge-Deployable Driver Drowsiness Detection: Closing the Lab-to-Real-World Generalization Gap with Lightweight Multimodal Fusion**

This repository contains the research implementation aligned with the above paper.

## 1. Project Overview

Driver drowsiness is an important road-safety concern. This project explores a lightweight multimodal deep-learning approach for detecting driver drowsiness using facial visual information and facial geometry features.

The implementation focuses on temporal behaviour, multimodal feature fusion, and cross-domain generalization to support real-world and edge-device deployment research.

## 2. Implemented Pipeline

The research implementation consists of the following components:

1. **MobileNetV2 visual encoder:** Extracts visual features from 224 × 224 RGB face crops.
2. **Facial geometry extraction:** Extracts five features per frame: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head yaw, pitch, and roll.
3. **Bidirectional cross-modal attention:** Combines visual and geometric information.
4. **Temporal modelling:** Uses a 30-frame Bidirectional LSTM (Bi-LSTM) to capture temporal patterns.
5. **Domain adaptation:** Uses a Gradient Reversal Layer (GRL) and domain classifier.
6. **Classification and evaluation:** Includes alert/drowsy classification, evaluation metrics, checkpoint saving, and benchmarking scripts.

**Important:** The paper's targets of ≤8 percentage-point cross-dataset performance degradation and ≥10 FPS on Raspberry Pi 4-class hardware are research targets, not measured results. No fabricated performance metrics are reported.

Actual dataset videos are not bundled with this repository. Obtain them from their official providers and follow their respective terms of use.

## 3. Technologies Used

- Python 3.11 (recommended)
- PyTorch
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Pandas
- PyYAML
- Scikit-learn

## 4. Project Structure

```text
driver-drowsiness-detection/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── manifests/
│   │   └── source_videos_template.csv
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   └── LABEL_MAPPING.md
│
├── results/
│   ├── README.md
│   └── results_template.csv
│
├── scripts/
│   ├── benchmark.py
│   ├── check_environment.py
│   ├── evaluate.py
│   ├── make_splits.py
│   ├── preprocess_video.py
│   ├── train.py
│   └── webcam_demo.py
│
├── src/
│   └── ddd/
│       ├── data.py
│       ├── geometry.py
│       ├── losses.py
│       ├── metrics.py
│       ├── model.py
│       └── utils.py
│
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

## 5. Installation and Setup

Python 3.11 is recommended.

### Step 1: Clone the repository

```bash
git clone https://github.com/priya10881/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Check the environment

```bash
python scripts/check_environment.py
```

CPU execution is supported. CUDA is optional. For CUDA, install the matching PyTorch build from:

https://pytorch.org/get-started/locally/

## 6. Dataset Access

Refer to the following documentation:

- `docs/DATASETS.md`
- `docs/LABEL_MAPPING.md`

The intended evaluation protocol is:

- **NTHU-DDD:** Training and held-out subject-independent testing.
- **YawDD:** Zero-shot external evaluation.
- **UTA-RLDD:** Zero-shot external evaluation.
- **MRL Eye:** Mentioned for eye-state pretraining; it is not silently mixed into driver-level sequence labels by this repository's sequence trainer.

The intended protocol does not use target-dataset fine-tuning for zero-shot evaluation.

### Dataset Manifest Format

Create a CSV file named:

`data/manifests/source_videos.csv`

Use the following format:

```csv
video_path,label,subject_id,domain
data/raw/NTHU-DDD/example.avi,1,sub001,NTHU-DDD
```

Labels:

- `0 = alert`
- `1 = drowsy`

Map each dataset's official labels only after reviewing its annotation definitions. Yawning should not automatically be treated as drowsiness.

## 7. Video Preprocessing

The preprocessing script extracts face crops and five-dimensional facial geometry features.

```bash
python scripts/preprocess_video.py --input-csv data/manifests/source_videos.csv --output-dir data/processed --sequence-length 30 --sample-fps 30
```

The script uses MediaPipe Face Mesh and generates:

- Face crops
- Five-dimensional geometry arrays
- `data/processed/manifest.csv`

Each window inherits the video-level label, which represents weak supervision. Use segment-level labels where available and document the label-mapping procedure.

## 8. Subject-Independent Data Splitting

Run:

```bash
python scripts/make_splits.py --manifest data/processed/manifest.csv --output-dir data/manifests/splits
```

Inspect the resulting subject IDs and ensure that there is no subject leakage between training and validation sets.

For strict source-only training, use NTHU-DDD rows for training and reserve YawDD and UTA-RLDD for external evaluation.

## 9. Model Training

Run the training script:

```bash
python scripts/train.py --train-csv data/manifests/splits/train.csv --val-csv data/manifests/splits/val.csv --config configs/default.yaml --output-dir checkpoints
```

### Optional Domain Adaptation

Unlabelled target-domain data can optionally be supplied for GRL-based domain adaptation:

```bash
python scripts/train.py --train-csv data/manifests/splits/train.csv --val-csv data/manifests/splits/val.csv --target-csv data/manifests/target_unlabelled.csv --config configs/default.yaml --output-dir checkpoints
```

Target CSV labels must be `-1`; only domain IDs are used for target samples.

Target test labels must not be used during domain adaptation.

## 10. Evaluation and Benchmarking

Evaluate the trained model on the NTHU-DDD test set:

```bash
python scripts/evaluate.py --csv data/manifests/nthudd_test.csv --checkpoint checkpoints/best.pt --output results/nthudd_test.json
```

Evaluate on YawDD:

```bash
python scripts/evaluate.py --csv data/manifests/yawdd_test.csv --checkpoint checkpoints/best.pt --output results/yawdd_zero_shot.json
```

Evaluate on UTA-RLDD:

```bash
python scripts/evaluate.py --csv data/manifests/utarldd_test.csv --checkpoint checkpoints/best.pt --output results/utarldd_zero_shot.json
```

Run the benchmark:

```bash
python scripts/benchmark.py --checkpoint checkpoints/best.pt --device cpu
```

For paper-related edge deployment claims, benchmarking should be performed on actual Raspberry Pi 4 (4 GB) or Jetson Nano hardware. Laptop CPU performance is only a proxy.

## 11. Webcam Demonstration

Run:

```bash
python scripts/webcam_demo.py --camera 0
```

**Note:** The webcam demonstration is an EAR/MAR geometric-cue baseline only. It does not represent the trained multimodal model.

## 12. Results and Reporting

Results and experiment documentation are maintained in the `results/` and `docs/` directories.

The repository distinguishes between:

- Research targets
- Experiment configurations
- Actual measured results

Performance claims should only be reported after running the corresponding experiments and benchmarks.

## 13. Limitations

This project is a research prototype and is not a certified vehicle safety system.

Potential limitations include:

- Facial landmark extraction may fail due to occlusion, sunglasses, poor lighting, or off-axis faces.
- Dataset annotation definitions and label semantics may differ.
- Video-level labels can introduce weak-supervision noise.
- Cross-dataset performance may vary due to differences in recording conditions and participant characteristics.
- Edge-device performance must be independently measured on the target hardware.

Further validation on held-out subjects, external datasets, and real embedded hardware is necessary before making deployment claims.

## 14. GitHub Repository

**Repository:**  
https://github.com/priya10881/driver-drowsiness-detection

The repository contains the research implementation, configuration files, documentation, preprocessing scripts, training scripts, and evaluation utilities.

Large datasets, processed crops, and model checkpoints are excluded from version control. Restricted dataset videos should never be uploaded.

---

**Research implementation for academic and experimental purposes.**