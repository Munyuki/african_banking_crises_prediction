markdown
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

### Feature Importance
![Feature Importance](<img width="542" height="460" alt="Feature Importance" src="https://github.com/user-attachments/assets/259c3dc8-6783-4d4c-8b28-1d811e8ace61" />
)

### Precision-Recall Curve
![Precision-Recall Curve](<img width="646" height="532" alt="Precision Recall Curve" src="https://github.com/user-attachments/assets/879a379c-ec88-4b5b-be3a-25886fd5d487" />
)

---

## Limitations

- Historical data (pre-2000) may not reflect modern financial systems
- Future crises may have different drivers (e.g., COVID-19, climate shocks)
- Data limited to African countries — patterns may differ globally

---

## How to Run This Project

1. **Clone or download** this repository
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
```

1. Open the notebook:
   ```bash
   jupyter notebook african_crises_project.ipynb
   ```
2. Run all cells

---

Future Improvements

- **Add recent data (2015–present)** — To test if patterns hold in modern economies
- **Experiment with time-series models (LSTM)** — To capture temporal crisis patterns
- **Build a simple dashboard** — For policymakers to monitor risk indicators

---

Connect with Me

· LinkedIn: [https://www.linkedin.com/in/nicodimus-munyuki-0832851b9]
· Email: [nicodimusmunyuki2@gmail.com]

---

Acknowledgments

Dataset compiled from historical economic records. Project completed as part of data science portfolio.
