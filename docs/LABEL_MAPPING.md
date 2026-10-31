# Label mapping

Model labels: `0=alert`, `1=drowsy`, `-1=unknown` (only for unlabelled target-domain training rows).

Before building manifests, inspect official annotations, define and document a mapping, exclude ambiguous labels, and report sample counts. Yawning alone is not ground truth for drowsiness. The preprocessor currently inherits one video-level label for every extracted window (weak supervision); use timestamp-level annotations if available. Keep target test labels out of training and adaptation.
