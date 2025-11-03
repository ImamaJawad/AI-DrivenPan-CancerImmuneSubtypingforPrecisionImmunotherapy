# 🧬 Pan-Cancer Immune Subtype Classification using RNA-seq Data

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20SVM%20%7C%20Neural%20Networks-green.svg)](https://scikit-learn.org/)

> **AI-Driven Precision Medicine**: Machine learning pipeline for classifying tumor immune microenvironments across 33 cancer types using TCGA RNA-seq data

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Clinical Implications](#clinical-implications)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#contact)

---

## 🎯 Overview

This project develops a **robust machine learning framework** to classify tumor samples into **6 immune subtypes** (C1-C6) based on gene expression profiles from **9,126 patients** across **33 cancer types** (TCGA PanCan Atlas). 

### **Why This Matters**
Immune subtyping enables:
- ✅ **Personalized immunotherapy selection** (e.g., predicting checkpoint inhibitor response)
- ✅ **Patient stratification** for clinical trials
- ✅ **Prognostic insights** based on tumor microenvironment
- ✅ **Drug discovery** targeting specific immune landscapes

### **The 6 Immune Subtypes (Thorsson et al., 2018)**
| Subtype | Key Characteristics | Clinical Relevance |
|---------|---------------------|-------------------|
| **C1** | Wound Healing | Angiogenic, Th2 dominance |
| **C2** | IFN-γ Dominant | Best prognosis, high CD8+ T cells |
| **C3** | Inflammatory | Th17, Th1 responses |
| **C4** | Lymphocyte Depleted | Immunosuppressed, M2 macrophages |
| **C5** | Immunologically Quiet | Low immune infiltration |
| **C6** | TGF-β Dominant | Stromal, poor prognosis |

---

## ✨ Key Features

### **1. Comprehensive Data Pipeline**
```
TCGA RNA-seq (20,531 genes) → Feature Engineering (440 immune genes) 
→ Preprocessing → Dimensionality Reduction → Classification
```

### **2. Advanced ML Models Compared**
- 🌳 **Random Forest** (ensemble learning)
- 🚀 **XGBoost** (gradient boosting, **83.3% accuracy**)
- 🔮 **SVM** (kernel methods, 81.8% accuracy)
- 🧠 **Neural Networks** (deep learning)
- 📊 **Logistic Regression** (baseline)

### **3. Dimensionality Reduction Techniques**
- **PCA** (Principal Component Analysis)
- **NMF** (Non-negative Matrix Factorization)
- **Kernel PCA** (non-linear relationships)
- **t-SNE** (visualization)

### **4. Rigorous Evaluation**
- Stratified 80/20 train-test split
- 3-fold cross-validation
- Hyperparameter tuning (GridSearchCV)
- Confusion matrices, precision, recall, F1-scores

---

## 📊 Dataset

### **Source**
- **TCGA PanCanAtlas**: 9,126 RNA-seq samples  
- **440 immune-signature genes** (Thorsson et al., 2018)
- **Balanced dataset**: 2,009 samples after stratification

### **Class Distribution (Balanced)**
| Subtype | Samples |
|---------|---------|
| C1 | 385 |
| C2 | 414 |
| C3 | 383 |
| C4 | 462 |
| C5 | 231 |
| C6 | 134 |

### **Preprocessing Steps**
1. ✅ Extracted 440 immune genes from 20,531 total genes
2. ✅ Filtered primary solid tumors (sample code: `01`)
3. ✅ Handled duplicates (BCR annotations)
4. ✅ Imputed missing values (mean imputation)
5. ✅ Normalized (Min-Max, StandardScaler)
6. ✅ Removed outliers (IQR method)

---

## 🔬 Methodology

### **Feature Engineering**
- **Gene Signatures**: 5 categories (S1-S5)
  - `S1`: Interferon-γ response
  - `S2`: Extracellular matrix/angiogenesis
  - `S3`: Immune cell infiltration
  - `S4`: B/T cell markers
  - `S5`: Cell cycle/proliferation

### **Dimensionality Reduction**
```python
# PCA (90% variance retention) → 94 components
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_full)

# NMF (100 components for interpretability)
nmf = NMF(n_components=100, init='nndsvda')
X_nmf = nmf.fit_transform(X_nonneg)
```

### **Model Training Pipeline**
```python
# XGBoost with GridSearchCV
param_grid = {
    'max_depth': [1, 2],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}
xgb_clf = GridSearchCV(XGBClassifier(), param_grid, cv=3)
```

---

## 📈 Results

### **Best Model: XGBoost on Full Dataset**
```
🏆 Overall Accuracy: 83.33%
```

#### **Per-Class Performance**
| Subtype | Precision | Recall | F1-Score | Interpretation |
|---------|-----------|--------|----------|----------------|
| **C1** | 0.82 | 0.86 | 0.84 | Excellent |
| **C2** | 0.90 | 0.92 | 0.91 | **Best** (strong IFN-γ signature) |
| **C3** | 0.83 | 0.82 | 0.82 | Very Good |
| **C4** | 0.80 | 0.79 | 0.80 | Good |
| **C5** | 0.88 | 0.91 | 0.89 | Excellent |
| **C6** | 0.65 | 0.56 | 0.60 | Challenging (small sample, overlap with C4) |

### **Key Findings**
✅ **C2 (IFN-γ Dominant)** easiest to classify (91% F1) → Clear CD8+ T cell signature  
✅ **C5 (Immunologically Quiet)** also performs well (89% F1)  
⚠️ **C6 (TGF-β Dominant)** hardest (60% F1) → Heterogeneous, overlaps with C4  

### **Model Comparison Across Feature Sets**
| Model | Full Dataset | PCA (94D) | NMF (138D) | t-SNE (3D) |
|-------|--------------|-----------|------------|------------|
| XGBoost | **83.3%** | 74.6% | 80.9% | 62.4% |
| SVM | 81.8% | 77.4% | 78.4% | 70.7% |
| Random Forest | 80.4% | 72.9% | 78.9% | 72.1% |
| Neural Network | 81.0% | 76.0% | 81.0% | 64.0% |

**Insight**: Full feature set outperforms dimensionality reduction (↑ 8.7% vs PCA), showing biological complexity cannot be overly compressed.

---

## 🚀 Installation

### **Prerequisites**
```bash
Python 3.10+
pip install -r requirements.txt
```

### **Dependencies**
```txt
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.3.0
xgboost==2.0.3
tensorflow==2.15.0
matplotlib==3.7.5
seaborn==0.12.2
sweetviz==2.3.1
```

### **Quick Start**
```bash
# Clone repository
git clone https://github.com/yourusername/immune-subtype-classification.git
cd immune-subtype-classification

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook immune_classification.ipynb
```

---

## 💻 Usage

### **1. Load Preprocessed Data**
```python
import pandas as pd

# Load balanced dataset
df = pd.read_csv("data/Final_csv_2009_samples.csv")

# Features (440 genes) + Target (Immune Subtype)
X = df.drop(columns=['Immune Subtype'])
y = df['Immune Subtype']
```

### **2. Train XGBoost Classifier**
```python
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
xgb_clf = XGBClassifier(
    max_depth=2, 
    learning_rate=0.1, 
    n_estimators=200
)
xgb_clf.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = xgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
```

### **3. Predict New Samples**
```python
# Load new RNA-seq data
new_patient = pd.read_csv("new_patient_expression.csv")

# Predict immune subtype
prediction = xgb_clf.predict(new_patient)
print(f"Predicted Subtype: {prediction[0]}")
```

---

## 📂 Project Structure

```
immune-subtype-classification/
│
├── data/
│   ├── PanCanAtlas_9126_samples.csv          # Raw TCGA data
│   ├── Final_csv_2009_samples.csv            # Balanced dataset
│   └── metadata.xlsx                          # Clinical annotations
│
├── notebooks/
│   ├── 01_data_preparation.ipynb              # Preprocessing
│   ├── 02_EDA_visualization.ipynb             # Exploratory analysis
│   ├── 03_feature_engineering.ipynb           # Dimensionality reduction
│   └── 04_model_training.ipynb                # Classification models
│
├── src/
│   ├── preprocessing.py                       # Data cleaning functions
│   ├── feature_selection.py                   # NMF/PCA/t-SNE
│   ├── models.py                              # XGBoost/SVM/NN
│   └── evaluation.py                          # Metrics, plots
│
├── results/
│   ├── confusion_matrices/
│   ├── feature_importance/
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🏥 Clinical Implications

### **For Oncologists**
✅ **Treatment Selection**: Predict response to checkpoint inhibitors (C2 responds best)  
✅ **Prognosis**: C2 = best survival, C6 = poorest outcomes  
✅ **Immunotherapy Resistance**: Identify C4/C6 patients needing combination therapy  

### **For Researchers**
✅ **Biomarker Discovery**: S5 genes (proliferation) correlate strongly with C5  
✅ **Drug Development**: Target TGF-β in C6 tumors  
✅ **Clinical Trials**: Stratify patients by immune subtype for better trial design  

### **Case Example**
```
🔬 Patient X: Lung cancer sample
   → Gene Expression → XGBoost Model → Predicted: C2 (IFN-γ Dominant)
   → Clinical Decision: Recommend anti-PD-1 therapy (high response rate in C2)
   → Expected Outcome: Improved progression-free survival
```

---

## 🔮 Future Work

### **Immediate Next Steps**
1. ⬜ **Multi-omics Integration**: Add mutations, methylation, proteomics  
2. ⬜ **External Validation**: Test on independent datasets (ICGC, GEO)  
3. ⬜ **Deep Learning**: Transformer models for gene expression  
4. ⬜ **Explainability**: SHAP values for model interpretability  

### **Long-Term Vision**
- 🎯 **Clinical Decision Support System**: Web app for real-time predictions  
- 🎯 **UAE Healthcare Integration**: Collaborate with hospitals (Dubai, Abu Dhabi)  
- 🎯 **Liquid Biopsy**: Extend to circulating tumor DNA (ctDNA)  
- 🎯 **Federated Learning**: Privacy-preserving multi-institutional training  

---

## 📚 References

### **Primary Literature**
1. **Thorsson, V. et al. (2018).** *The Immune Landscape of Cancer*. Immunity, 48(4), 812-830.e14.  
   [DOI: 10.1016/j.immuni.2018.03.023](https://doi.org/10.1016/j.immuni.2018.03.023)

2. **TCGA Research Network.** *The Cancer Genome Atlas Program*.  
   [genomic.cancer.gov/tcga](https://www.cancer.gov/tcga)

### **Machine Learning Methods**
- Chen, T. & Guestrin, C. (2016). *XGBoost: Reliable Large-scale Tree Boosting*. KDD '16.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825-2830.

### **Biological Context**
- Galon, J. & Bruni, D. (2019). *Approaches to treat immune hot, altered and cold tumours*. Nature Reviews, 18, 197-218.
---

## 🙏 Acknowledgments

- **TCGA Research Network** for publicly available data
- **Thorsson et al. (2018)** for immune subtype definitions
- **Scikit-learn & XGBoost developers**
- **Academic supervisors** and collaborators

---

**⭐ If this project helped you, please star the repository!**

```
Built with ❤️ for advancing precision oncology through AI
```
