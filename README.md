# Vehicle Insurance Classifier with Imbalanced Data

This project predicts whether an existing health insurance customer will be interested in the company's vehicle insurance. It compares SMOTE oversampling and NearMiss undersampling across three scikit-learn classifiers.

Project work for a machine learning course at the University of Bologna, 2021.

## Dataset

The notebook describes the data as provided by an insurance company that sells health insurance to its customers. No public source is named.

- `Train dataset.csv`: 382,154 rows, 12 columns, including the target `Response`. It is imbalanced: 319,553 rows have `Response` 0 and 62,601 have `Response` 1 (about 16%). There are no missing values.
- `Test dataset.csv`: 78,273 rows with the same columns but no `Response`. It is not used in the notebook.

The features fall into three groups: demographics (gender, age, region code), vehicle (vehicle age, damage), and policy (premium, sales channel).

| Column | Values |
|---|---|
| `id` | int, unique identity |
| `Gender` | Male / Female |
| `Age` | int |
| `Driving_License` | 0/1 |
| `Region_Code` | int |
| `Previously_Insured` | 0/1 |
| `Vehicle_Age` | `< 1 Year`, `1-2 Year`, `> 2 Years` |
| `Vehicle_Damage` | Yes/No |
| `Annual_Premium` | int |
| `Policy_Sales_Channel` | int |
| `Vintage` | int |
| `Response` | 0/1, target (interested in vehicle insurance) |

## Approach

1. **Exploratory analysis.** Check for missing values, then plot the distributions of age, gender, driving license, previous insurance, vehicle age, vehicle damage, premium and the target.
2. **Preprocessing.** One-hot encode `Gender`, `Vehicle_Age` and `Vehicle_Damage` with `pd.get_dummies`, which gives 15 features. The other columns, including `id`, are used unscaled. The data is split 70/30 into train and test (`random_state=1`): 267,507 training rows and 114,647 test rows.
3. **Resampling (training split only).**
   - SMOTE (`random_state=2`) oversamples class 1 from 43,907 to 223,600 rows, matching class 0.
   - NearMiss (imbalanced-learn defaults) undersamples class 0 from 223,600 to 43,907 rows.
4. **Models.** Logistic regression, random forest and gradient boosting with default settings, trained on each resampled set. Logistic regression is also trained on the original imbalanced data as a reference.
5. **Tuning.** `GridSearchCV` tunes gradient boosting over learning rate (0.01, 0.02, 0.03, 0.04), subsample (0.1, 0.2, 0.5, 0.9) and max depth (4, 6, 8, 10), with `n_estimators` fixed at 10. Selection uses accuracy on the resampled training data.
   - SMOTE best: learning rate 0.03, max depth 10, subsample 0.2 (cross-validated accuracy 0.884).
   - NearMiss best: learning rate 0.03, max depth 6, subsample 0.1 (cross-validated accuracy 0.915).

## Results

The table reports results on the 30% hold-out test split of `Train dataset.csv` (114,647 rows: 95,953 class 0 and 18,694 class 1), taken from the notebook outputs. `accuracy` comes from `accuracy_score`; the other columns come from `classification_report`. "(1)" and "(0)" mark the class.

| Resampling | Model | accuracy | precision (1) | recall (1) | f1-score (1) | f1-score (0) | macro avg f1-score |
|---|---|---|---|---|---|---|---|
| None | Logistic regression | 0.834 | 0.38 | 0.03 | 0.05 | 0.91 | 0.48 |
| SMOTE | Logistic regression | 0.657 | 0.27 | 0.65 | 0.38 | 0.76 | 0.57 |
| SMOTE | Random forest | 0.876 | 0.65 | 0.52 | 0.58 | 0.93 | 0.75 |
| SMOTE | Gradient boosting | 0.860 | 0.57 | 0.60 | 0.58 | 0.92 | 0.75 |
| SMOTE | Gradient boosting, tuned | 0.872 | 0.63 | 0.53 | 0.57 | 0.92 | 0.75 |
| NearMiss | Logistic regression | 0.341 | 0.17 | 0.81 | 0.29 | 0.39 | 0.34 |
| NearMiss | Random forest | 0.716 | 0.35 | 0.89 | 0.50 | 0.80 | 0.65 |
| NearMiss | Gradient boosting | 0.706 | 0.34 | 0.88 | 0.49 | 0.79 | 0.64 |
| NearMiss | Gradient boosting, tuned | 0.717 | 0.35 | 0.88 | 0.50 | 0.80 | 0.65 |

- Without resampling, logistic regression predicts almost only class 0 (recall 0.03 for class 1).
- SMOTE with random forest gives the highest accuracy and, tied with untuned gradient boosting, the highest class-1 f1-score. This is also the notebook's conclusion.
- With the tree models, NearMiss raises class-1 recall to 0.88 to 0.89, but class-1 precision falls to 0.34 to 0.35.

Limitations: the `id` column is used as a feature, and features are not scaled (logistic regression on the NearMiss data stops at the iteration limit). Results come from a single train/test split without fixed seeds for the models, so reruns will vary slightly.

## How to run

1. Install the dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter`. The notebook was run with scikit-learn 0.24.2 and imbalanced-learn 0.8.1. Newer versions may show warnings or need small changes.
2. Start Jupyter from the repository root and open `Project work in ML.ipynb`. The notebook reads `Train dataset.csv` from the working directory.
3. The first cell (`!pip install imblearn`) can be skipped if imbalanced-learn is already installed. The SMOTE grid search took about 8 minutes in the original run.

## Files

| Path | Contents |
|---|---|
| `Project work in ML.ipynb` | Full analysis: EDA, preprocessing, resampling, models, tuning |
| `Train dataset.csv` | Labeled data used for training and testing |
| `Test dataset.csv` | Unlabeled data (not used) |
| `Project work for ML.pptx` | Presentation slides |
