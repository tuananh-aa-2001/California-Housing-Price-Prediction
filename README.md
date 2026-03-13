# 🏠 California Housing Price Prediction

A complete end-to-end machine learning regression project that predicts California median house values using the classic [Aurélien Géron](https://github.com/ageron/handson-ml2) dataset. The pipeline covers every stage from raw data acquisition through model fine-tuning, evaluation, and persistence.

---

## 📁 Project Structure

```
Housing Price Prediction/
│
├── housingprice.py            # Main script — optimized ML pipeline (~420 lines)
│
├── datasets/                  # Auto-created on first run
│   ├── housing.tgz            # Downloaded compressed archive
│   ├── housing.csv            # Extracted flat CSV (fallback path)
│   └── housing/
│       └── housing.csv        # Primary dataset (20,640 rows × 10 cols)
│
├── final_housing_pipeline_complete.pkl # Serialised unified pipeline (prep + select + predict)
│
├── housing_geography.png      # EDA: Geo scatter plot of house values
├── correlation_heatmap.png    # EDA: Pearson correlation matrix heatmap
├── feature_importances.png    # Model: Top-15 feature importances bar chart
├── predictions_vs_actual.png  # Eval: Predicted vs actual scatter plot
├── residual_analysis.png      # Eval: Residuals vs predicted + distribution
│
└── README.md                  # This file
```

---

## 🔄 Step-by-Step Workflow

The project is structured into **7 key phases** for maximum efficiency and modularity.

### Step 1 — Data Acquisition & Setup
Downloads and extracts the California housing dataset automatically.

### Step 2 — Stratified Train/Test Split
Uses `StratifiedShuffleSplit` on income categories to ensure the test set (20%) is representative of the overall population's wealth distribution.

### Step 3 — Exploratory Data Analysis (EDA)
Generates geographical and correlation visualizations using modular helper functions.

### Step 4 — Data Preparation
Builds a preprocessing pipeline including:
- **Numerical**: Median imputation, feature engineering (ratios), and standard scaling.
- **Categorical**: One-hot encoding for `ocean_proximity`.

### Step 5 — Consolidated Research & Tuning (Speed Optimized) 🚀
This is the heart of the optimization. Instead of separate training steps, it uses a **Global Search Strategy**:
- **Subsampling**: Uses only **20% of training data** during initial exploration to identify the best model archetype 5x faster.
- **Unified Pipeline**: Chains preparation, top-feature selection, and the estimator into one object.
- **Randomized Search**: Explores **Random Forest, Gradient Boosting, and SVR** configurations simultaneously with reduced CV folds (3) for rapid iteration.
- **Final Fit**: Re-trains the single winning configuration on the **full training set** for maximum accuracy.

### Step 6 — Final Evaluation & Visualization
Applies the winner to the held-out test set and generates performance plots (Residuals, Feature Importances).

### Step 7 — Save & Wrap Up
Serializes the entire unified pipeline into `final_housing_pipeline_complete.pkl` for one-line deployment.

---

### Step 11 — Example Prediction

Five samples are drawn from the test set and run through the **unified pipeline** in one step:

```python
sample_predictions = final_pipeline.predict(sample_data)
```

Output shows predicted price, actual price, and the absolute error for each sample.

---

## ⚙️ How to Run

### Prerequisites

Install required packages (Python 3.8+):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### Run the full pipeline

```bash
python housingprice.py
```

The script will:
1. Download the dataset automatically.
2. Train all models (Randomised and Grid Search may take several minutes).
3. Save the unified `final_housing_pipeline_complete.pkl`.
4. Save the five PNG visualisation files.

### Load the saved model for inference

```python
import joblib

# Load the single unified pipeline file
full_pipeline = joblib.load('final_housing_pipeline_complete.pkl')

# Predict directly on raw data matching the original feature set
predictions = full_pipeline.predict(X_new)
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Stratified split on `income_cat` | Prevents income-distribution skew in train/test sets |
| Median imputation for `total_bedrooms` | Robust to outliers; ~200 missing values in the dataset |
| Engineered ratio features | Raw counts (rooms, bedrooms) correlate with each other; ratios per household carry more information |
| `StandardScaler` on numerical features | Required for Linear/Ridge regression; doesn't hurt tree models |
| `TopFeatureSelector` | Automatically prunes less relevant features to reduce model complexity |
| Data Subsampling | Uses a 20% subset for exploration to identify best models 5x faster |
| Consolidated Search | Merges model tuning and prep exploration into a single global phase |
| Unified Pipeline | Encapsulates preparation and prediction for one-click deployment |
| `joblib` for model persistence | More efficient than `pickle` for large NumPy arrays inside scikit-learn models |

---

## 📊 Dataset Source

- **Origin**: California census data, 1990
- **Hosted by**: [Aurélien Géron — Hands-On ML2 GitHub](https://github.com/ageron/handson-ml2)
- **Direct URL**: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz`
- **Rows**: 20,640 &nbsp;|&nbsp; **Columns**: 10 &nbsp;|&nbsp; **Missing values**: ~207 in `total_bedrooms`
