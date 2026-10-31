# Driver Drowsiness Detection — Research Implementation

Aligned with the DA3 paper: **Real-Time, Edge-Deployable Driver Drowsiness Detection: Closing the Lab-to-Real-World Generalization Gap with Lightweight Multimodal Fusion**.

Implemented pipeline:
1. MobileNetV2 visual encoder (224×224 RGB face crops)
2. Five geometry features per frame: EAR, MAR, head yaw, pitch, roll
3. Bidirectional cross-modal attention
4. 30-frame Bi-LSTM temporal model
5. Gradient Reversal Layer (GRL) + domain classifier
6. Alert/drowsy classifier, metrics, checkpoints and benchmark script

**Important:** the paper's ≤8 percentage-point cross-dataset degradation and ≥10 FPS Raspberry Pi 4-class performance are targets, not measured results. No fabricated metrics are included. Actual dataset videos are not bundled; obtain them from the official providers and follow their terms.

## Install (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```
CPU works; CUDA is optional. For CUDA, install the matching PyTorch build from https://pytorch.org/get-started/locally/.

## Dataset access
See `docs/DATASETS.md` and `docs/LABEL_MAPPING.md`. Intended protocol: train on NTHU-DDD; held-out subject-independent NTHU-DDD test; zero-shot YawDD and UTA-RLDD tests (no target fine-tuning). MRL Eye is mentioned in the paper for eye-state pretraining; this repo's sequence trainer does not silently mix it into driver-level labels.

Create `data/manifests/source_videos.csv` with:
```csv
video_path,label,subject_id,domain
data/raw/NTHU-DDD/example.avi,1,sub001,NTHU-DDD
```
Labels: `0=alert`, `1=drowsy`. Map each dataset's official labels only after reviewing its annotation definitions. Yawning is not automatically drowsiness.

## Preprocess to 30-frame windows
```bash
python scripts/preprocess_video.py --input-csv data/manifests/source_videos.csv --output-dir data/processed --sequence-length 30 --sample-fps 30
```
The script uses MediaPipe Face Mesh, writes face crops and 5-D geometry arrays, and creates `data/processed/manifest.csv`. Each window inherits the video-level label (weak supervision); use segment-level labels if available and document your mapping.

## Subject-independent split
```bash
python scripts/make_splits.py --manifest data/processed/manifest.csv --output-dir data/manifests/splits
```
Inspect the resulting subject IDs and ensure no leakage. For strict source-only training, train on NTHU-DDD rows only; keep YawDD/UTA-RLDD for external evaluation.

## Train
```bash
python scripts/train.py --train-csv data/manifests/splits/train.csv --val-csv data/manifests/splits/val.csv --config configs/default.yaml --output-dir checkpoints
```
Optional unlabelled target-domain data for GRL:
```bash
python scripts/train.py --train-csv data/manifests/splits/train.csv --val-csv data/manifests/splits/val.csv --target-csv data/manifests/target_unlabelled.csv --config configs/default.yaml --output-dir checkpoints
```
Target CSV labels must be `-1`; only domain IDs are used for target samples. Do not use target test labels during adaptation.

## Evaluate / benchmark
```bash
python scripts/evaluate.py --csv data/manifests/nthudd_test.csv --checkpoint checkpoints/best.pt --output results/nthudd_test.json
python scripts/evaluate.py --csv data/manifests/yawdd_test.csv --checkpoint checkpoints/best.pt --output results/yawdd_zero_shot.json
python scripts/evaluate.py --csv data/manifests/utarldd_test.csv --checkpoint checkpoints/best.pt --output results/utarldd_zero_shot.json
python scripts/benchmark.py --checkpoint checkpoints/best.pt --device cpu
```
For paper claims, benchmark on actual Raspberry Pi 4 (4 GB) / Jetson Nano. Laptop CPU is only a proxy.

## Webcam
`python scripts/webcam_demo.py --camera 0` is an EAR/MAR geometric cue baseline only, not the trained multimodal model.

## GitHub
```bash
git init
git add .
git commit -m "Initial driver drowsiness research implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
Large data, crops and checkpoints are ignored. Never upload restricted videos.

## Limitations
Research prototype only; not a certified vehicle safety system. Landmark extraction may fail with occlusion, sunglasses, poor lighting or off-axis faces. Dataset label semantics differ. Validate on held-out subjects, external datasets and real embedded hardware before reporting results.
