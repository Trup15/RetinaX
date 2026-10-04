# Data Specification & Acquisition Guide — RetinaX

This document provides the exact attributes, schemas, folder structures, and download commands required for the RetinaX project to operate with real medical data.

---

## 1. The Three Benchmark Datasets

### Dataset 1: APTOS 2019 Blindness Detection
- **Source:** Aravind Eye Hospital, Tamil Nadu, India.
- **Role in RetinaX:** Primary training, internal validation, and temperature scaling ($T = 1.38$) calibration.
- **Size:** $3,662$ training images with verified specialist labels; $1,928$ unlabelled test images.
- **Attributes in `train.csv`:**
  | Column Name | Type | Allowed Values | Description |
  |---|---|---|---|
  | `id_code` | `string` | e.g. `000c1434d8d7` | Unique image filename (without `.png` extension) |
  | `diagnosis` | `integer` | `0, 1, 2, 3, 4` | Ordinal clinical DR stage according to ICDR scale |

- **Class Distribution:**
  - `0` (No DR): $1,805$ samples ($49.3\%$)
  - `1` (Mild NPDR): $370$ samples ($10.1\%$)
  - `2` (Moderate NPDR): $999$ samples ($27.3\%$)
  - `3` (Severe NPDR): $193$ samples ($5.3\%$)
  - `4` (Proliferative DR): $295$ samples ($8.1\%$)

- **How to Get It:**
  ```bash
  # Using Kaggle CLI (requires kaggle.json in ~/.kaggle/)
  kaggle competitions download -c aptos2019-blindness-detection
  unzip aptos2019-blindness-detection.zip -d datasets/aptos2019/
  ```

---

### Dataset 2: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Source:** Eye Clinic in Nanded, Maharashtra, India (Kowa VX-10alpha digital camera, $50^\circ$ FOV).
- **Role in RetinaX:** External Indian generalization validation + Quantitative pixel-level lesion validation (Objective 4).
- **Size:** $516$ fundus images ($4288 \times 2848$ resolution).
- **Sub-datasets & Attributes:**
  1. **Disease Grading (`IDRiD_Disease_Grading.csv`):**
     | Column Name | Type | Allowed Values | Description |
     |---|---|---|---|
     | `Image name` | `string` | e.g. `IDRiD_042.jpg` | Image filename |
     | `Retinopathy grade` | `integer` | `0, 1, 2, 3, 4` | Ground-truth DR grade |
     | `Risk of macular edema` | `integer` | `0, 1, 2` | Diabetic Macular Edema (DME) risk |
  2. **Pixel-Level Lesion Ground Truth Masks (`.tif` binary images):**
     - `1. Microaneurysms/`: Binary masks with value `255` at capillary outpouchings.
     - `2. Haemorrhages/`: Binary masks of blot and flame intraretinal bleeding.
     - `3. Hard Exudates/`: Binary masks of yellowish waxy protein/lipid deposits.
     - `4. Soft Exudates/`: Binary masks of cotton wool nerve fiber ischemic patches.

- **How to Get It:**
  - Official IEEE Dataport: [https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) (DOI: `10.21227/H25W98`).
  - Or via Kaggle:
    ```bash
    kaggle datasets download -d mariaherrerot/idrid-dataset
    unzip idrid-dataset.zip -d datasets/idrid/
    ```

---

### Dataset 3: DDR (Diabetic Retinopathy Dataset)
- **Source:** Multi-center cohort across 147 hospitals in China.
- **Role in RetinaX:** Training the MobileNetV3-Small quality gate + Stress-testing broad cross-dataset domain shift.
- **Size:** $13,673$ fundus images.
- **Attributes in `dr_labels.txt`:**
  | Column Name | Type | Allowed Values | Description |
  |---|---|---|---|
  | `image_name` | `string` | e.g. `007-0001-000.jpg` | Image filename |
  | `label` | `integer` | `0, 1, 2, 3, 4, 5` | `0-4`: DR severity; `5`: **Ungradable** (severe blur, media opacity, poor lighting) |

- **Why It Is Required:** The `label = 5` ungradable subset ($1,150$ images) provides the positive training samples for the `MobileNetV3-Small` Quality Gate to learn optical blur and cataract rejection.

---

## 2. Directory Layout Expected by the Backend

```
datasets/
+-- aptos2019/
|   +-- train_images/          # 3662 .png files
|   +-- train.csv
+-- idrid/
|   +-- original_images/       # 516 .jpg files
|   +-- ground_truth/
|   |   +-- disease_grading.csv
|   |   +-- 1_microaneurysms/  # .tif masks
|   |   +-- 2_haemorrhages/    # .tif masks
|   |   +-- 3_hard_exudates/   # .tif masks
|   |   +-- 4_soft_exudates/   # .tif masks
+-- ddr/
|   +-- images/                # 13673 .jpg files
|   +-- annotations/
|       +-- train.txt
|       +-- valid.txt
|       +-- test.txt
```

---

## 3. Data Preprocessing Pipeline Contract

Before being passed to PyTorch models, all fundus images must undergo:
1. **Circular FOV Masking:** Identify pupil radius $R$ and center $(c_x, c_y)$. Crop tight bounding box $[c_x - R, c_y - R, 2R, 2R]$ removing black borders.
2. **Resizing:** Standardized bilinear interpolation to $512 \times 512 \times 3$.
3. **Color Normalization:**
   ```python
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
   ```
4. **Data Augmentation (Training Only via Albumentations):**
   - Horizontal & Vertical Flips ($p = 0.5$)
   - Random Affine Rotation ($\pm 180^\circ$)
   - Color Jitter (Brightness $\pm 0.1$, Contrast $\pm 0.1$)
   - CLAHE (Contrast Limited Adaptive Histogram Equalization, $p = 0.3$)
