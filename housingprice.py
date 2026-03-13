import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform, reciprocal, expon
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- HELPER CLASSES & FUNCTIONS ---

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def indices_of_top_features(importances, k):
    return np.sort(np.argsort(importances)[::-1][:k])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_features(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

def plot_geography(housing, save_path="housing_geography.png"):
    plt.figure(figsize=(12, 8))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", figsize=(12,8),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 sharex=False)
    plt.legend()
    plt.title("California Housing Prices (Geographical Distribution)")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Geographical plot saved as '{save_path}'")

def plot_correlations(housing, save_path="correlation_heatmap.png"):
    corr_matrix = housing.select_dtypes(include=[np.number]).corr()
    print("\nCorrelation with median_house_value:")
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Correlation heatmap saved as '{save_path}'")

def plot_feature_importances(importances, attributes, save_path="feature_importances.png"):
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances[:15])), importances[sorted_indices[:15]])
    plt.title("Feature Importances (Top 15)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(range(len(importances[:15])), 
               [attributes[i] for i in sorted_indices[:15]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Feature importances plot saved as '{save_path}'")

def plot_residuals(y_true, y_pred, save_path_scatter="predictions_vs_actual.png", 
                   save_path_residuals="residual_analysis.png"):
    # Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values ($)")
    plt.ylabel("Predicted Values ($)")
    plt.title("Predictions vs Actual Values (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path_scatter, dpi=100, bbox_inches='tight')
    plt.show()
    
    # Residual Analysis
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.tight_layout()
    plt.savefig(save_path_residuals, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Visualizations saved as '{save_path_scatter}' and '{save_path_residuals}'")

# 1. DATA ACQUISITION & SETUP
print("="*60)
print("HOUSING PRICES PREDICTION - REGRESSION PROBLEM")
print("="*60)

def fetch_housing_data():
    """Download and extract the housing dataset"""
    # Create directory if it doesn't exist
    if not os.path.isdir("datasets"):
        os.makedirs("datasets")
    
    # Download the data
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
    tgz_path = os.path.join("datasets", "housing.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    
    # Extract the data
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path="datasets")
    housing_tgz.close()
    
    print("Data downloaded and extracted successfully!")

# Fetch the data
fetch_housing_data()

# Load the data
def load_housing_data():
    # The tgz archive extracts housing.csv directly into datasets/
    csv_path = "datasets/housing/housing.csv"
    if not os.path.exists(csv_path):
        csv_path = "datasets/housing.csv"
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(f"\nDataset shape: {housing.shape}")
print(f"\nFirst 5 rows of the dataset:")
print(housing.head())

print(f"\nDataset info:")
print(housing.info())

print(f"\nSummary statistics:")
print(housing.describe())

# 2. CREATE A TEST SET
print("\n" + "="*60)
print("CREATING TRAIN AND TEST SETS")
print("="*60)

# First, create an income category for stratified sampling
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print("\nIncome category distribution:")
print(housing["income_cat"].value_counts().sort_index())

# Stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Check the income category proportions
print("\nIncome category proportions in full dataset:")
print(housing["income_cat"].value_counts() / len(housing))

print("\nIncome category proportions in stratified train set:")
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

print("\nIncome category proportions in stratified test set:")
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Remove the income_cat attribute to get the data back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Create a copy for exploration
housing = strat_train_set.copy()

# 3. EXPLORE THE DATA (EDA)
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Visualize geographical data
plot_geography(housing)

# Visualize correlations
plot_correlations(housing)

# 4. PREPARE THE DATA FOR MACHINE LEARNING
print("\n" + "="*60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("="*60)

# Separate predictors and labels
housing = strat_train_set.drop("median_house_value", axis=1)  # Drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# Define numerical and categorical attributes
num_attribs = list(housing.select_dtypes(include=[np.number]).columns)
cat_attribs = ["ocean_proximity"]

print(f"\nNumerical attributes: {num_attribs}")
print(f"Categorical attributes: {cat_attribs}")


# Create pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

# Full pipeline for both numerical and categorical data
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# Prepare the data
housing_prepared = full_pipeline.fit_transform(housing)
print(f"\nPrepared data shape: {housing_prepared.shape}")

# 5. CONSOLIDATED MODEL RESEARCH & TUNING
print("\n" + "="*60)
print("CONSOLIDATED MODEL RESEARCH & TUNING (SPEED OPTIMIZED)")
print("="*60)

# Create a master dictionary to track all CV results
research_results = {}

# 5.1 Speed Optimization: Subsample the data for exploratory search
# We'll use 20% of the data to find the best archetype and rough params
exploration_sample_size = 0.2
exploration_indices = np.random.choice(len(housing), int(len(housing) * exploration_sample_size), replace=False)
housing_exploration = housing.iloc[exploration_indices]
labels_exploration = housing_labels.iloc[exploration_indices]

print(f"\nUsing {exploration_sample_size*100:.0f}% of training data for exploration ({len(housing_exploration)} samples)")

# Define numerical features for indexing
feature_names = num_attribs + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']

# 5.2 Define the full unified pipeline structure
unified_pipeline = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances=None, k=5)),
    ("prediction", RandomForestRegressor(random_state=42))
])

# We need baseline feature importances once.
print("Calculating base feature importances for selection logic (on full set)...")
forest_baseline = RandomForestRegressor(random_state=42)
forest_baseline.fit(housing_prepared, housing_labels)
base_importances = forest_baseline.feature_importances_
unified_pipeline.named_steps["feature_selection"].feature_importances = base_importances

best_score = float('inf')
best_params = None
best_archetype_name = None

# Archetypes with faster search settings
archetypes = [
    {
        "name": "Random Forest",
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "prediction__n_estimators": randint(low=50, high=150),
            "prediction__max_features": randint(low=1, high=8),
            "feature_selection__k": list(range(5, 12))
        }
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "prediction__n_estimators": randint(low=50, high=150),
            "prediction__learning_rate": [0.05, 0.1, 0.2],
            "feature_selection__k": list(range(5, 12))
        }
    },
    {
        "name": "SVR",
        "model": SVR(),
        "params": {
            "prediction__kernel": ["linear", "rbf"],
            "prediction__C": reciprocal(20, 50000),
            "prediction__gamma": expon(scale=1.0),
            "feature_selection__k": list(range(5, 12))
        }
    }
]

for archetype in archetypes:
    print(f"\n--- Exploring {archetype['name']} ---")
    unified_pipeline.set_params(prediction=archetype["model"])
    
    # Speed Optimization: Fewer iterations (10) and fewer folds (3)
    rnd_search = RandomizedSearchCV(unified_pipeline, param_distributions=archetype["params"],
                                    n_iter=10, cv=3, scoring='neg_mean_squared_error',
                                    random_state=42, n_jobs=-1, verbose=1)
    rnd_search.fit(housing_exploration, labels_exploration)
    
    current_rmse = np.sqrt(-rnd_search.best_score_)
    research_results[archetype["name"]] = current_rmse
    print(f"Best Subsampled CV RMSE: ${current_rmse:,.2f}")
    
    if current_rmse < best_score:
        best_score = current_rmse
        best_params = rnd_search.best_params_
        best_archetype_name = archetype["name"]
        final_pipeline = rnd_search.best_estimator_

print(f"\nWinner: {best_archetype_name}")
print(f"Best search RMSE (on subset): ${best_score:,.2f}")

# 5.4 FINAL TRAINING: Train the winning configuration on the FULL training set
print(f"\nRefitting the winning {best_archetype_name} model on the FULL training set...")
final_pipeline.fit(housing, housing_labels)
print("Final training complete.")

# 6. EVALUATE & VISUALIZE FINAL MODEL
print("\n" + "="*60)
print("EVALUATING FINAL OPTIMIZED MODEL ON TEST SET")
print("="*60)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# End-to-end prediction
final_predictions = final_pipeline.predict(X_test)

# Calculate metrics
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print(f"\nFinal Statistics:")
print(f"  RMSE     : ${final_rmse:,.2f}")
print(f"  MAE      : ${final_mae:,.2f}")
print(f"  R² Score : {final_r2:.4f}")

# Visualizations
plot_residuals(y_test, final_predictions)

# Get feature names for importance plot (using forest logic if winner is forest-like)
if hasattr(final_pipeline.named_steps['prediction'], 'feature_importances_'):
    # We need attributes for the SELECTED features
    cat_encoder = full_pipeline.named_transformers_['cat']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    all_attr_names = feature_names + cat_one_hot_attribs
    
    selected_indices = final_pipeline.named_steps['feature_selection'].feature_indices_
    selected_attr_names = [all_attr_names[i] for i in selected_indices]
    
    plot_feature_importances(final_pipeline.named_steps['prediction'].feature_importances_, 
                             selected_attr_names)

# 7. SAVE & WRAP UP
print("\n" + "="*60)
print("PROJECT PERSISTENCE AND CONCLUSION")
print("="*60)

import joblib
joblib.dump(final_pipeline, 'final_housing_pipeline_complete.pkl')
print("\nOptimization complete. Unified pipeline saved as 'final_housing_pipeline_complete.pkl'")

# Final sample test
sample_data = X_test.iloc[:5]
sample_labels = y_test.iloc[:5]
preds = final_pipeline.predict(sample_data)

print("\nSample Results (Predictions vs Actual):")
for p, a in zip(preds, sample_labels):
    print(f"  Predicted: ${p:,.0f} | Actual: ${a:,.0f} | Error: ${abs(p-a):,.0f}")

print("\n" + "="*60)
print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
print("="*60)