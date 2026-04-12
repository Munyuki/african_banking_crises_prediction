# Predicting Banking Crises in African Economies

**Author:** Nicodimus Munyuki  
**Tools:** Python, pandas, scikit-learn, matplotlib, seaborn

---

## Overview

Banking crises can devastate economies — wiping out savings, collapsing businesses, and triggering recessions. This project builds a machine learning model to **predict banking crises before they happen** using macroeconomic indicators across 13 African countries from 1860 to 2014.

---

## Dataset

- **Source:** Historical macroeconomic data from 13 African countries
- **Countries:** Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia, Zimbabwe
- **Time period:** 1860 – 2014
- **Size:** 1,059 observations, 14 features
- **Target variable:** `banking_crisis` (crisis vs. no_crisis)

### Class Distribution (Imbalanced)
| Class | Percentage |
|-------|------------|
| No Crisis | 91.1% |
| Crisis | 8.9% |

This imbalance reflects real-world rarity of banking crises but makes prediction challenging.

---

## Approach

### 1. Data Preprocessing
- One-hot encoding for categorical variables
- Feature scaling using `StandardScaler`
- Train-test split (80/20) with stratification

### 2. Handling Class Imbalance
- **SMOTE** (Synthetic Minority Over-sampling) to balance training data
- **Class weights** for Random Forest

### 3. Models Tested
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 93% | 0.968 |
| K-Nearest Neighbors | 96% | 0.938 |
| Random Forest | **98%** | **0.988** |

### 4. Hyperparameter Tuning
Used GridSearchCV to find optimal parameters for all three models.

---

## Key Results

- **Best model:** Random Forest with **98% accuracy** and **0.988 ROC-AUC**
- **Precision for crisis class:** 94% (few false alarms)
- **Recall for crisis class:** 79% (catches 15 out of 19 actual crises)

### Confusion Matrix (Random Forest)
```

[[192   1]   ← No crisis predicted correctly
[  4  15]]  ← Crises predicted correctly

```

### Top 3 Predictors of Banking Crises
1. **Systemic crisis** — Existing financial instability
2. **Year** — Temporal patterns in crisis occurrence
3. **Independence** — Political/economic autonomy

---

## Visualizations

### ROC Curve
<img width="426" height="413" alt="ROC Curve" src="https://github.com/user-attachments/assets/5487d084-179c-47f2-9f46-f38828aa0823" />

The Random Forest model achieves an AUC of 0.99 on the ROC curve, meaning it excels at distinguishing between crisis and non-crisis years. A value this close to 1.0 indicates very few false positives and false negatives

### Feature Importance
<img width="670" height="532" alt="Feature_Importance" src="https://github.com/user-attachments/assets/0db9bbf4-9a5c-4f89-8286-083dee5495bb" />

Feature importance confirms that systemic crises and year are the strongest predictors, followed by political independence and inflation. The presence of country-specific features (Egypt, South Africa) suggests that geographic context matters.

### Precision-Recall Curve
<img width="646" height="532" alt="Precision Recall Curve" src="https://github.com/user-attachments/assets/879a379c-ec88-4b5b-be3a-25886fd5d487" />

The Precision-Recall curve is especially informative for imbalanced data (only 8.9% crises). Random Forest achieves the highest Average Precision (AP = 0.936), meaning it maintains both high precision and recall across thresholds. This is critical in a crisis prediction setting — you want to avoid false alarms while still catching real crises.

---

## Limitations

- Historical data (pre-2000) may not reflect modern financial systems
- Future crises may have different drivers (e.g., COVID-19, climate shocks)
- Data limited to African countries — patterns may differ globally

---

## Future Improvements

- **Add recent data (2015–present)** — To test if patterns hold in modern economies
- **Experiment with time-series models (LSTM)** — To capture temporal crisis patterns
- **Build a simple dashboard** — For policymakers to monitor risk indicators


