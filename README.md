# Tamil Handwritten Character Recognition  

This repository implements multiple **deep learning architectures** for recognizing **Tamil handwritten characters**.  
The dataset consists of **156 character classes**, preprocessed into **64×64 grayscale images**.  

The project benchmarks different neural network architectures — **Custom CNN, DenseNet, GoogLeNet (HCCR-inspired), and Capsule CNN** — and compares their performance using consistent preprocessing, training, and evaluation strategies.  

---
# Results Summary
| Model      | Accuracy (%) |
|------------|--------------|
| CNN        | 96.94        |
| DenseNet   | 96.87        |
| HCCR       | 96.14        |
| CapsNet    | 93.10        |

# Features
- **Custom Dataset Loader** – Uses CSVs mapping filenames to labels.  
- **Image Preprocessing** – Resizing, grayscale conversion, normalization, and augmentations (rotation, affine transforms, scaling).  
- **Models Implemented**
  - `implementation_cnn.py` → Custom 5-block CNN  
  - `densenetmodel.py` → DenseNet-121 (adapted for 64×64 grayscale)  
  - `hccrmodel.py` → GoogLeNet (HCCR-inspired with Inception blocks)  
  - `capsnetcnn.py` → Capsule CNN with squashing nonlinearity  
- **Training Utilities**
  - **Optimizer:** Adam (`lr=1e-3` or `1e-4`)  
  - **Loss:** CrossEntropyLoss  
  - **Learning Rate Scheduler:** `ReduceLROnPlateau` – halves learning rate when validation loss plateaus  
  - **Early Stopping:** Stops training after 10 patience epochs  
- **Evaluation & Visualization**
  - Accuracy and classification reports (precision, recall, F1)  
  - Confusion matrix with top-10 confused classes heatmap  
  - Loss & accuracy curves across epochs  
  - Prediction samples with confidence scores  

---

# Tech Stack
- **Frameworks:** PyTorch, Torchvision  
- **Data & Evaluation:** NumPy, Pandas, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Progress Tracking:** tqdm  

---

# Training & Evaluation Workflow

## Data Loading
- Reads `FileNames` and `Ground Truth` from CSV files.  
- Custom `HandwritingDataset` class wraps image loading and transforms.  

## Training
- Each script defines a `train_model()` function.  
- Training settings:
  - Epochs: **200**  
  - Batch size: **32**  
  - Optimizer: **Adam**  
  - Scheduler: **ReduceLROnPlateau** (reduces LR if validation loss stalls)  
  - Early stopping: **patience = 10**  

## Evaluation
- Best model checkpoint auto-saved (e.g., `bestcnn_model_scheduler.pth`).  
- Evaluation outputs:
  - Test accuracy  
  - Classification report  
  - Confusion matrix  
  - Top misclassifications  
  - Prediction confidence scores  

---

# Sample Predictions

## CNN Model (96.94%)
| Sample | True Label | Predicted Label | Confidence |
|--------|------------|-----------------|------------|
| 0      | 94         | 94              | 1.0000     |
| 1      | 81         | 81              | 1.0000     |
| 2      | 91         | 91              | 1.0000     |
| 3      | 44         | 44              | 1.0000     |
| 4      | 120        | 120             | 0.9989     |
| 5      | 134        | 134             | 0.9309     |
| 6      | 14         | 15              | 0.6577     |
| 7      | 148        | 148             | 1.0000     |
| 8      | 30         | 30              | 1.0000     |
| 9      | 31         | 31              | 1.0000     |

## DenseNet Model (96.87%)
| Sample | True Label | Predicted Label | Confidence |
|--------|------------|-----------------|------------|
| 0      | 94         | 94              | 1.0000     |
| 1      | 81         | 81              | 1.0000     |
| 2      | 91         | 91              | 1.0000     |
| 3      | 44         | 44              | 1.0000     |
| 4      | 120        | 120             | 0.9983     |
| 5      | 134        | 134             | 0.6897     |
| 6      | 14         | 14              | 1.0000     |
| 7      | 148        | 148             | 1.0000     |
| 8      | 30         | 30              | 1.0000     |
| 9      | 31         | 31              | 1.0000     |

## HCCR Model (96.14%)
| Sample | True Label | Predicted Label | Confidence |
|--------|------------|-----------------|------------|
| 0      | 94         | 94              | 0.9999     |
| 1      | 81         | 81              | 0.9999     |
| 2      | 91         | 91              | 1.0000     |
| 3      | 44         | 44              | 1.0000     |
| 4      | 120        | 120             | 0.9972     |
| 5      | 134        | 134             | 0.9105     |
| 6      | 14         | 14              | 0.7637     |
| 7      | 148        | 148             | 1.0000     |
| 8      | 30         | 30              | 0.9999     |
| 9      | 31         | 31              | 1.0000     |

## CapsNet CNN Model (93.10%)
| Sample | True Label | Predicted Label | Confidence |
|--------|------------|-----------------|------------|
| 0      | 94         | 94              | 0.9997     |
| 1      | 81         | 81              | 1.0000     |
| 2      | 91         | 91              | 1.0000     |
| 3      | 44         | 44              | 1.0000     |
| 4      | 120        | 120             | 0.8329     |
| 5      | 134        | 134             | 0.9634     |
| 6      | 14         | 15              | 0.8758     |
| 7      | 148        | 148             | 1.0000     |
| 8      | 30         | 30              | 1.0000     |
| 9      | 31         | 31              | 1.0000     |

# Dataset Source and Reference Paper
- Dataset: https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database
- N. Shaffi and F. Hajamohideen, "uTHCD: A New Benchmarking for Tamil Handwritten OCR," in IEEE Access, vol. 9, pp. 101469-101493, 2021, doi: 10.1109/ACCESS.2021.3096823.
- For more details about the dataset, Please refer the above paper.
