# Data Specification & Local Test Generation — RetinaX

This document provides both the **official dataset acquisition steps** and a **zero-dependency test data generator** so you can develop and test the entire backend immediately without downloading multi-gigabyte datasets.

---

## 1. Quick Start: Zero-Download Synthetic Test Data Generator
Run this script to immediately create a realistic local test dataset in `datasets/test_cohort/` (takes 2 seconds, no Kaggle account needed):

```python
# Save as: scripts/generate_test_cohort.py
import os
import cv2
import numpy as np

def generate_synthetic_fundus(filename, dr_stage=0, is_blurry=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Background orange-red retina
    cv2.circle(img, (256, 256), 220, (15, 45, 180), -1)
    
    # Optic disc (yellowish)
    cv2.circle(img, (160, 260), 28, (20, 160, 240), -1)
    
    # Macula (darker red)
    cv2.circle(img, (310, 260), 30, (8, 20, 100), -1)
    
    # Blood vessels
    cv2.line(img, (160, 260), (340, 120), (5, 10, 80), 3)
    cv2.line(img, (160, 260), (340, 400), (5, 10, 80), 3)
    
    # Inject lesions according to DR stage
    if dr_stage >= 1: # Microaneurysms
        for _ in range(8):
            rx, ry = np.random.randint(200, 360), np.random.randint(180, 340)
            cv2.circle(img, (rx, ry), 2, (0, 0, 140), -1)
            
    if dr_stage >= 2: # Hard Exudates (yellow spots)
        for _ in range(12):
            ex, ey = np.random.randint(220, 380), np.random.randint(200, 320)
            cv2.circle(img, (ex, ey), 3, (120, 230, 250), -1)
            
    if dr_stage >= 3: # Blot Hemorrhages
        for _ in range(6):
            hx, hy = np.random.randint(180, 380), np.random.randint(160, 360)
            cv2.circle(img, (hx, hy), 7, (0, 0, 160), -1)

    if is_blurry:
        img = cv2.GaussianBlur(img, (31, 31), 15)

    cv2.imwrite(filename, img)

if __name__ == "__main__":
    generate_synthetic_fundus("datasets/test_cohort/normal_eye.jpg", dr_stage=0)
    generate_synthetic_fundus("datasets/test_cohort/mild_dr.jpg", dr_stage=1)
    generate_synthetic_fundus("datasets/test_cohort/moderate_dr.jpg", dr_stage=2)
    generate_synthetic_fundus("datasets/test_cohort/severe_dr.jpg", dr_stage=3)
    generate_synthetic_fundus("datasets/test_cohort/blurry_ungradable.jpg", dr_stage=2, is_blurry=True)
    print("Generated 5 realistic test images in datasets/test_cohort/")
```

Run it via:
```bash
python scripts/generate_test_cohort.py
```

---

## 2. Official Production Datasets (Kaggle & IEEE)

### Dataset 1: APTOS 2019 Blindness Detection
- **Role:** Primary training & temperature scaling calibration ($T = 1.38$).
- **Size:** $3,662$ labeled train images, $1,928$ test images.
- **Attributes in `train.csv`:**
  - `id_code`: Image filename without extension (e.g. `000c1434d8d7`).
  - `diagnosis`: Integer DR grade ($0$: Normal, $1$: Mild, $2$: Moderate, $3$: Severe, $4$: Proliferative).
- **Download Command:**
  ```bash
  kaggle competitions download -c aptos2019-blindness-detection
  unzip aptos2019-blindness-detection.zip -d datasets/aptos2019/
  ```

### Dataset 2: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Role:** External Indian generalization validation + Ground-truth lesion segmentation (Objective 4).
- **Size:** $516$ fundus photographs with pixel-level lesion binary `.tif` masks.
- **Sub-folders:**
  - `1_microaneurysms/`: Microaneurysm binary masks.
  - `2_haemorrhages/`: Intraretinal hemorrhage binary masks.
  - `3_hard_exudates/`: Hard exudate binary masks.
  - `4_soft_exudates/`: Cotton wool spot binary masks.
- **Download:** [IEEE Dataport (DOI: 10.21227/H25W98)](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) or:
  ```bash
  kaggle datasets download -d mariaherrerot/idrid-dataset
  unzip idrid-dataset.zip -d datasets/idrid/
  ```

### Dataset 3: DDR (Diabetic Retinopathy Dataset)
- **Role:** Training the MobileNetV3-Small quality gate (label `5` = ungradable/blur/cataract).
- **Size:** $13,673$ fundus images across 147 hospitals.
- **Download:** Kaggle `ddr-dataset` or GitHub DDR benchmark repo.
