# Dataset links and usage

Do not redistribute dataset files in this repository. Obtain access from providers and follow their current license, research-use and citation terms.

- **NTHU-DDD** (primary training dataset): https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- **YawDD** (external evaluation): https://qualinet.github.io/databases/video/yawdd_a_yawning_detection_dataset/
- **YawDD IEEE DataPort listing**: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
- **UTA-RLDD** (external evaluation): https://sites.google.com/view/utarldd/home
- **MRL Eye Dataset** (eye-state pretraining mentioned in paper): https://mrl.cs.vsb.cz/eyedataset

Suggested local folders: `data/raw/NTHU-DDD`, `data/raw/YawDD`, `data/raw/UTA-RLDD`, `data/raw/MRL-Eye`. MRL Eye images are not consumed by the sequence-level trainer in this repository. YawDD focuses on yawning behavior; yawning must not be automatically equated with drowsiness.
