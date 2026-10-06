# Obesity Level Classification Using Machine Learning

A machine learning project that classifies individuals into **seven obesity levels** based on personal data, eating habits and physical condition. The project covers the full pipeline: exploratory analysis, preprocessing, feature engineering, model comparison, hyperparameter tuning, ensemble learning, a data-leakage study, and a comparison against published research.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Leakage Analysis: BMI and Weight](#leakage-analysis-bmi-and-weight)
- [Comparison With Previous Studies](#comparison-with-previous-studies)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Obesity is a major global health problem linked to chronic diseases such as diabetes and heart disease. Accurate classification of obesity levels can support diagnosis and help guide treatment plans and recommendation systems in digital health.

This project investigates how well different machine learning models can predict obesity level, which features matter most, and how much of a model's accuracy depends on features that are almost a direct function of the target (BMI and weight).

**Target classes:**

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

---

## Dataset

**Estimation of Obesity Levels Based on Eating Habits and Physical Condition**, from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5H31Z).

It contains personal and behavioral attributes such as age, height, weight, vegetable consumption, number of main meals, water intake, physical activity frequency, screen time, transportation type, snacking, alcohol consumption, smoking, calorie monitoring, high-calorie food consumption and family history of obesity.

---

## Methodology

### 1. Exploratory Data Analysis
- Histograms for numerical features and a bar chart of the target classes (the classes are roughly balanced, with Obesity Type I slightly more frequent)
- Scatter plot of height vs. weight by obesity level
- Frequency tables and statistical tests comparing categorical features against obesity level (transportation, snacking, alcohol, calorie monitoring, smoking, high-calorie foods, family history)
- Correlation matrix of numerical features

Key observations:
- Frequent consumption of high-calorie foods and a family history of obesity are clearly associated with higher obesity levels.
- Monitoring calorie intake appears to be a protective factor.
- Smoking and snacking show no clear relationship with obesity level.

### 2. Preprocessing
- Renamed unclear column names to readable ones
- No missing values were found
- Removed 5 duplicate rows
- Outlier analysis using the IQR method (1.5 and 3 multipliers) plus manual inspection of extreme values. The detected outliers were judged to be realistic values, so the data was kept unchanged.

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `BMI` | Weight (kg) / height (m)² |
| `Risk Factor` | Score from 1 to 4 based on four behavioral features: smoking, high-calorie food, calorie monitoring and exercise frequency |
| `Nutrition Score` | Score based on vegetable intake, number of meals and water intake |

### 4. Encoding and Splitting
- Label encoding for the target
- One-hot encoding for categorical features inside `ColumnTransformer` pipelines (one variant for linear models, one for tree-based models)
- 80/20 train/test split

---

## Models

| Role | Model |
|------|-------|
| Baseline | Logistic Regression |
| Compared | SVM, Decision Tree, Random Forest, XGBoost, LightGBM |
| Tuning | `RandomizedSearchCV` (10 iterations, 3-fold cross-validation) on LightGBM |
| Feature selection | Automatic feature selection on the tuned model |
| Ensemble | Soft `VotingClassifier` combining LightGBM and SVM |

Evaluation uses accuracy, precision, recall, F1-score, confusion matrices (heatmaps) and per-class classification reports.

---

## Results

- LightGBM achieved the best single-model performance when all features were used.
- After hyperparameter tuning, LightGBM reached a test accuracy of **0.9569**.
- After automatic feature selection, test accuracy improved to **0.9617**.
- The soft-voting ensemble (LightGBM + SVM) achieved the highest test accuracy, although SVM was the more stable of the two models.

**Final model performance:**

| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 96.52% | 96.14% | 96.29% | 96.14% |

---

## Leakage Analysis: BMI and Weight

Since BMI and weight are almost a direct function of the target, the project tests how much the models depend on them using two scenarios:

1. **Remove both** weight and BMI
2. **Remove BMI only**, keep weight

Tested on SVM, Decision Tree, Random Forest and LightGBM.

| Model | Without BMI and weight | With weight only |
|-------|------------------------|------------------|
| SVM | Lower accuracy, more errors | Improved |
| Decision Tree | Weak class separation | Improved, but below BMI-based performance |
| Random Forest | Acceptable but weaker | Improved |
| LightGBM | Clear drop in performance | Improved |

**Takeaway:** Weight on its own boosts performance noticeably and may carry information that the BMI ratio does not capture. At the same time, relying fully on these features can hide the influence of behavioral and personal factors, so a balance between the two feature types is recommended to avoid data leakage.

---

## Comparison With Previous Studies

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **This project (final model)** | 96.52% | 96.14% | 96.29% | 96.14% | n/a |
| Study 1: Logistic Regression + RFE (Gozukara Bag et al., 2023) | 98.99% | 99.83% | 99.01% | 98.99% | 99.96% |
| Study 2: XGBoost (Buani and Nuraeni, 2023) | 96.00% | 96.00% | 96.00% | 96.00% | n/a |
| Study 3: Random Forest (Dirik, 2023) | 95.78% | ~96-99% | ~96% | ~96% | ~1.00 |

The final model is competitive with the published results. It outperforms Studies 2 and 3 on accuracy, while Study 1 reports higher scores.

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Main libraries:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
scipy
jupyter
```

---

## Usage

1. Download the dataset from the [UCI repository](https://doi.org/10.24432/C5H31Z) and place it in the project folder (for example `data/`).
2. Open the notebook and run the cells in order:

```bash
jupyter notebook
```

The notebook follows the same order as the sections above: EDA, preprocessing, feature engineering, model training, tuning, ensemble, and evaluation.

---

## Future Work

- Expand the sample size and test on real-world data from different sources and regions to improve generalization
- Add clinical indicators such as blood pressure, blood sugar, triglycerides and cholesterol
- Try deep neural networks and hybrid models

---

## References

1. Gozukara Bag, H. G., et al. (2023). *Estimation of obesity levels through the proposed predictive approach based on physical activity and nutritional habits.* Diagnostics, 13(18), 2949. https://doi.org/10.3390/diagnostics13182949
2. Buani, D. C. P., and Nuraeni, N. (2023). *Application of XGB classifier for obesity rate prediction.* Jurnal Riset Informatika, 6(1). https://doi.org/10.34288/jri.v6i1.260
3. Dirik, M. (2023). *Application of machine learning techniques for obesity prediction: A comparative study.* Journal of Complexity in Health Sciences, 6(2). https://doi.org/10.21595/chs.2023.23193
4. *Estimation of Obesity Levels Based On Eating Habits and Physical Condition* [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z

---

## Disclaimer

This project is for educational and research purposes only and must not be used for medical diagnosis.

## License

Add a license of your choice (for example MIT) before publishing.
