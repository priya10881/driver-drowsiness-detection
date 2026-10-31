# Paper-aligned experiments

- Source training and validation: NTHU-DDD with subject-independent partitions.
- Source test: held-out NTHU-DDD subjects.
- External tests: YawDD and UTA-RLDD zero-shot, without fine-tuning.
- Inputs: 30 frames per sequence; 224×224 face crop; geometry `[EAR,MAR,yaw,pitch,roll]`.
- Metrics: accuracy, drowsy precision/recall/F1, ROC-AUC, confusion matrix, latency and FPS.
- Baselines in the paper: EAR/MAR threshold-only, frame-wise MobileNetV2, CNN-LSTM without geometry, proposed model.
- Ablations: remove geometry; remove GRL; shared/separate eye encoders; 15/30/45-frame windows; FP32/INT8.
- Report `source_test_accuracy - target_accuracy` as percentage-point gap separately for each target dataset.
- Record software versions, seed, dataset release, label mapping, subject IDs, hardware and checkpoint hash.
- ≤8 pp cross-domain degradation and ≥10 FPS Raspberry Pi 4-class are success targets, not guaranteed outcomes.

## Important protocol distinction

The paper describes zero-shot external testing and also describes feeding target-domain samples to the domain classifier. These are distinct settings:
- **Strict zero-shot:** source-only training; no target-domain samples are used at training time.
- **Unsupervised domain adaptation:** source labelled data plus unlabelled target adaptation subjects are used for GRL domain loss; final target test subjects are disjoint.

Report both separately if both are run. Do not call the second setting zero-shot.
