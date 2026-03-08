# Literature Review

## Skeletal Injury Diagnosis with CNN
**A Two-Stage Deep Learning Approach for Enhanced Diagnostic Accuracy**

# Research Gaps in Existing Systems

Despite advances in deep learning, several challenges remain unresolved.

## 1. Imaging Variability
Different X-ray acquisition conditions cause significant variation in medical images.

Challenges include:
- Different imaging angles affecting fracture appearance
- Variations in patient positioning
- Inconsistent image quality due to different machines

These factors require models that generalize well across diverse clinical environments.

---

## 2. Class Imbalance and Rare Fracture Types

Medical datasets often contain **uneven distribution of fracture categories**.

Issues include:
- Rare fractures being underrepresented in training data
- Models becoming biased toward majority classes
- Increased risk of false negatives for minority fracture types

---

## 3. Hairline Fracture Detection

Hairline fractures are extremely subtle and often span only **2–5 pixels** in X-ray images.

Challenges include:
- Difficulty distinguishing fractures from noise
- Requirement for deeper feature extraction
- Importance for early treatment planning

---

## 4. Anatomical Overlap

Bones often overlap in radiographic images, especially around joints.

Examples include:
- Elbow joint overlaps
- Shoulder joint overlaps
- Wrist and hand bone intersections

These overlapping structures can hide fracture lines and reduce detection accuracy when using a single generic model.

---

# Related Work

Several deep learning architectures have been explored for fracture detection.

| Architecture | Performance | Dataset |
|-------------|-------------|--------|
| Faster R-CNN | 70% precision | 3,067 X-ray images |
| YOLO v2 | 75.3% precision | MURA Dataset |
| Dilated Convolutional Feature Pyramid Network | 82.1% accuracy | 3,842 X-ray images |

These models demonstrate promising performance but often struggle with generalization and subtle fracture detection.

---

# Proposed Approach

To overcome existing limitations, we propose a **Two-Stage Detection Framework**.

## Stage 1: Bone Type Classification

The first stage identifies the anatomical region of the X-ray image.

Steps involved:

1. **Input Processing**
   - X-ray image uploaded via a graphical interface
   - Resized to **224 × 224 pixels**
   - Pixel normalization between **0 and 1**

2. **Data Augmentation**
   - Horizontal flipping applied
   - Excessive rotation avoided to maintain anatomical realism

3. **ResNet50 Classification**
   - Residual neural network extracts deep features
   - Softmax layer predicts probabilities for:
     - Elbow
     - Hand
     - Shoulder

Output: Bone type label and confidence score.

---

## Stage 2: Bone-Specific Fracture Detection

Based on the bone classification from Stage 1, a specialized model is selected.

### Model Training

Separate models are trained for each bone type:

| Bone Type | Training Images |
|----------|----------------|
| Elbow | 5,396 images |
| Hand | 6,003 images |
| Shoulder | 8,936 images |

Each model:

- Uses a **ResNet50 backbone**
- Learns bone-specific fracture patterns
- Performs **binary classification**
  - Fractured
  - Normal

Output includes fracture prediction and confidence score.

---

# Addressing Research Gaps

The proposed system addresses previously identified limitations.

## Hairline Fractures

ResNet50's deep architecture captures fine-grained features using residual connections, preserving edge information necessary for detecting tiny fracture lines.

---

## Anatomical Overlap

Training **bone-specific models** allows the system to learn normal anatomical overlaps unique to each region.

Examples:
- Ulna–radius overlaps in the elbow
- Metacarpal overlaps in the hand

---

## Class Imbalance

Separating models by bone type prevents dominant categories from overwhelming minority fracture classes.

For example:

- Hand fractures: 27.9%
- Shoulder fractures: 49.7%

Training models separately ensures balanced learning.

---

## Augmentation Overfitting

Only clinically realistic augmentations are applied.

Applied:
- Horizontal flipping

Avoided:
- Extreme rotations
- Artificial distortions

This ensures the model learns real fracture features rather than augmentation artifacts.

---

# Dataset Description

The system uses the **MURA (Musculoskeletal Radiographs) Dataset** developed by Stanford University.

Dataset statistics:

| Bone | Normal | Fractured |
|------|--------|-----------|
| Elbow | 3,160 | 2,236 |
| Hand | 4,330 | 1,673 |
| Shoulder | 4,496 | 4,440 |

Total images: **20,335 X-rays**

Class distribution:
- Normal: 59%
- Fractured: 41%

---

# References

1. Yıldırım, R., Singh, A., Sharma, K. (2024). Bone Fracture Detection with CNN-AlexNet. IEEE.  
https://ieeexplore.ieee.org/document/10722699

2. Bhatia, S., Gupta, M., Khan, R. (2020). Bone Fracture Detection using CNN. IEEE.  
https://ieeexplore.ieee.org/document/9087067

3. Yıldırım, O., Öz, S., Subasi, A. (2023). Skeletal Fracture Detection with Deep Learning.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC10606060/

4. Tanzi, L., Vezzetti, E., Moos, S. (2020). X-Ray Bone Fracture Classification using Deep Learning.  
https://www.mdpi.com/2076-3417/10/4/1507

5. Thian, Y. L., Li, Y., Ng, K. H. (2019). CNNs for Automated Fracture Detection.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC8017412/

6. Ju, R. Y., Lin, C. H., Wang, P. S. (2023). YOLO-based Fracture Detection.  
https://www.nature.com/articles/s41598-023-47460-7

7. Tahir, A., Ali, S., Ahmad, M. (2024). Ensemble Deep Learning Model for Fracture Detection.  
https://www.sciencedirect.com/science/article/pii/S0009926024004197

8. Bagaria, R., Wadhwani, S., Kaur, M. (2021). Bone Fracture Detection using SVM.  
https://www.sciencedirect.com/science/article/abs/pii/S0030402621015825

9. Hoover, R., Miller, D., Chen, Z. (2025). Pre-trained Under Noise: Robust Bone Fracture Detection.  
https://arxiv.org/abs/2507.09731

10. Chen, H., Wang, Y., Li, Z. (2020). Anatomy-Aware Siamese Network for Fracture Detection.  
https://arxiv.org/abs/2007.01464