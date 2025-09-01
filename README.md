


# Air Quality Classification using Tree Ensembles

## Introduction for Editors

This repository accompanies the submitted manuscript and provides the complete Python implementation used for the air quality classification experiments. The purpose of this repository is to ensure transparency and reproducibility of the results reported in the paper. All code is included in a single script (`air_quality_models.py`), along with clear instructions for setup, data preparation, and execution.  

The methodology applies **Synthetic Minority Oversampling Technique (SMOTE)** to address class imbalance and evaluates multiple tree-ensemble models, including Random Forest, Balanced Random Forest, Shallow Random Forest, and Extra Trees. Each model is configured with the hyperparameters specified in the manuscript, and evaluation metrics are produced for both test data and cross-validation.  

Editors and reviewers can run the script directly with a properly formatted dataset (Excel file with a `Classification` column) to reproduce the reported metrics, plots, and best-model selection. This repository is therefore intended as a transparent and verifiable companion to the manuscript.  

---

## 1) Environment Setup

**Python version**: 3.9 – 3.12 recommended  

Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies include:

* pandas
* numpy
* scikit-learn
* imbalanced-learn
* matplotlib
* seaborn
* openpyxl

---

## 2) Data

* The dataset must be provided in **Excel format (.xlsx)**.
* It must include a column named **`Classification`** with labels:

  * `"Good"`, `"Moderate"`, `"Unhealthy"`

The script maps these to numeric values:

```python
{'Good': 0, 'Moderate': 1, 'Unhealthy': 2}
```

**How to set your file path:**
Inside `air_quality_models.py`, update this line:

```python
df = pd.read_excel('Model_air_quality_data.xlsx')
```

Replace `'Model_air_quality_data.xlsx'` with your actual filename or path.

> ⚠️ Note: The dataset is **not included** in this repository due to size/privacy.

---

## 3) How to Run

Once your data file is ready, run:

```bash
python air_quality_models.py
```

The script will:

1. Split the dataset into **70% training** and **30% test** sets (stratified, random\_state=42).
2. Apply **SMOTE** to balance the training data.
3. Train and evaluate four models:

   * Random Forest (RF)
   * Balanced Random Forest (BRF)
   * Shallow Random Forest (SRF)
   * Extra Trees (ERF)
4. Report metrics for each model:

   * Accuracy
   * Weighted F1 Score
   * Weighted Precision
   * Weighted Recall
   * Confusion Matrix
   * Classification Report
5. Perform **5-fold Cross-Validation accuracy** on the original data (without SMOTE in CV).
6. Identify the **best model by test-set accuracy**.
7. Plot and display:

   * Class distribution before/after SMOTE
   * Evaluation metrics for each model
   * Feature importance plots
   * Best-model performance bar chart

---

## 4) Models & Hyperparameters

The models are configured as follows:

| Model         | Trees | Max Depth | Min Samples Leaf | Class Weight | Criterion | Bootstrap | Random State |
| ------------- | :---: | --------- | :--------------: | ------------ | --------- | --------- | ------------ |
| Random Forest |  300  | None      |         1        | None         | Gini      | True      | 42           |
| Balanced RF   |  300  | None      |         1        | Balanced     | Gini      | True      | 42           |
| Shallow RF    |  100  | 3         |         1        | None         | Gini      | True      | 42           |
| Extra Trees   |  200  | None      |         1        | None         | Gini      | True      | 42           |

> Note: For Extra Trees, `bootstrap=True` is explicitly set (default is False in scikit-learn).

---

## 5) Repository Structure

```
.
├─ air_quality_models.py        # Main Python script (full code)
├─ README.md                    # Project documentation
├─ requirements.txt             # Python dependencies
├─ .gitignore                   # Files to ignore (cache, data, venv, etc.)
├─ data/                        # (optional) place your Excel file here (not committed)
└─ outputs/                     # (optional) figures or logs if you save them
```

---

## 6) Outputs

When you run the script, you will see:

* Printed metrics in the console (Accuracy, F1, Precision, Recall)
* Confusion matrices and classification reports per model
* Bar charts for:

  * Class distribution before/after SMOTE
  * Evaluation metrics per model
  * Feature importances per model
  * Best-model metrics

Figures are displayed using matplotlib; you can optionally save them by adding:

```python
plt.savefig('outputs/<filename>.png', dpi=200, bbox_inches='tight')
```

---

## 7) Reproducibility Checklist

* Train/test split uses `random_state=42`
* Models use `random_state=42`
* SMOTE applied only to the **training set**
* Weighted metrics reported for imbalanced data
* 5-fold CV performed on the **original un-resampled data**

---

## 8) Citation


* If you use this repository, please cite both the manuscript and this GitHub repository.
```

