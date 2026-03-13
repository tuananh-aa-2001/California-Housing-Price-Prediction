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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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
plt.figure(figsize=(12, 8))
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(12,8),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.title("California Housing Prices (Geographical Distribution)")
plt.savefig("housing_geography.png", dpi=100, bbox_inches='tight')
plt.show()
print("Geographical plot saved as 'housing_geography.png'")

# Look for correlations (numeric columns only — pandas 2.x drops non-numeric columns from corr())
corr_matrix = housing.select_dtypes(include=[np.number]).corr()
print("\nCorrelation with median_house_value:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Visualize correlations with a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlations")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=100, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved as 'correlation_heatmap.png'")

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

# Custom transformer to add combined attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Assuming X is a numpy array with columns in the original order
        # rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

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

# 5. TRAIN AND EVALUATE MODELS
print("\n" + "="*60)
print("TRAINING AND EVALUATING MODELS")
print("="*60)

# Dictionary to store model performances
model_performances = {}

def train_and_evaluate(model, X, y, model_name, cv=10):
    """
    Fit a regression model, report train-set RMSE, run k-fold
    cross-validation, and store the mean CV RMSE for comparison.

    Parameters
    ----------
    model       : sklearn estimator
    X           : prepared feature matrix (numpy array)
    y           : target Series / array
    model_name  : str — label used in print output and results dict
    cv          : int — number of CV folds (default 10)

    Returns
    -------
    fitted model
    """
    # --- Fit ---
    model.fit(X, y)

    # --- Training-set RMSE ---
    train_preds = model.predict(X)
    train_rmse  = np.sqrt(mean_squared_error(y, train_preds))
    print(f"\n{model_name} RMSE on training set: ${train_rmse:,.2f}")

    # --- Cross-validation ---
    cv_scores      = -cross_val_score(model, X, y,
                                      scoring="neg_mean_squared_error", cv=cv)
    cv_rmse_scores = np.sqrt(cv_scores)
    print(f"{model_name} CV Scores:")
    print(f"  Scores : {cv_rmse_scores}")
    print(f"  Mean   : {cv_rmse_scores.mean():.4f}")
    print(f"  Std Dev: {cv_rmse_scores.std():.4f}")

    # --- Store for later comparison ---
    model_performances[model_name] = cv_rmse_scores.mean()

    return model

# Define all candidate models
candidates = [
    ("Linear Regression",  LinearRegression()),
    ("Decision Tree",       DecisionTreeRegressor(random_state=42)),
    ("Random Forest",       RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting",   GradientBoostingRegressor(n_estimators=100,
                                                       learning_rate=0.1,
                                                       random_state=42)),
]

# Train and evaluate every model with a single unified call
trained_models = {}
for name, model in candidates:
    print("\n" + "-"*40)
    print(f"Training {name}...")
    trained_models[name] = train_and_evaluate(
        model, housing_prepared, housing_labels, name
    )

# Convenience references used later in fine-tuning
forest_reg = trained_models["Random Forest"]

# 6. FINE-TUNE THE BEST MODEL
print("\n" + "="*60)
print("FINE-TUNING THE BEST MODEL")
print("="*60)

# Based on the performances, let's fine-tune Random Forest (usually performs well)
print("\nPerforming Grid Search for Random Forest...")

# 6.1 Grid Search
param_grid = [
    {'n_estimators': [50, 100, 200], 'max_features': [4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [50, 100], 'max_features': [4, 6]},
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs=-1)
grid_search.fit(housing_prepared, housing_labels)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {np.sqrt(-grid_search.best_score_):,.2f}")

# 6.2 Analyze feature importance
feature_importances = grid_search.best_estimator_.feature_importances_

# Get feature names
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + ['rooms_per_household', 'population_per_household', 
                            'bedrooms_per_room'] + cat_one_hot_attribs

# Sort feature importances
sorted_indices = np.argsort(feature_importances)[::-1]
print("\nTop 10 Most Important Features:")
for i in range(10):
    print(f"{i+1}. {attributes[sorted_indices[i]]}: {feature_importances[sorted_indices[i]]:.4f}")

# Visualize feature importances
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances[:15])), feature_importances[sorted_indices[:15]])
plt.title("Feature Importances (Top 15)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(range(len(feature_importances[:15])), 
           [attributes[i] for i in sorted_indices[:15]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=100, bbox_inches='tight')
plt.show()
print("Feature importances plot saved as 'feature_importances.png'")

# 7. EVALUATE ON TEST SET
print("\n" + "="*60)
print("EVALUATING FINAL MODEL ON TEST SET")
print("="*60)

# Prepare the test data
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

# Make predictions
final_predictions = final_model.predict(X_test_prepared)

# Calculate metrics
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print(f"\nFinal Model Performance on Test Set:")
print(f"RMSE: ${final_rmse:,.2f}")
print(f"MAE: ${final_mae:,.2f}")
print(f"R² Score: {final_r2:.4f}")

# Calculate confidence interval
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
ci_lower = np.sqrt(np.mean(squared_errors) - stats.t.ppf((1 + confidence) / 2, len(squared_errors) - 1) * 
                   np.std(squared_errors) / np.sqrt(len(squared_errors)))
ci_upper = np.sqrt(np.mean(squared_errors) + stats.t.ppf((1 + confidence) / 2, len(squared_errors) - 1) * 
                   np.std(squared_errors) / np.sqrt(len(squared_errors)))
print(f"\n{confidence*100:.0f}% Confidence Interval for RMSE: [${ci_lower:,.2f}, ${ci_upper:,.2f}]")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values ($)")
plt.ylabel("Predicted Values ($)")
plt.title("Predictions vs Actual Values (Test Set)")
plt.tight_layout()
plt.savefig("predictions_vs_actual.png", dpi=100, bbox_inches='tight')
plt.show()
print("Predictions vs actual plot saved as 'predictions_vs_actual.png'")

# Residual analysis
residuals = y_test - final_predictions
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(final_predictions, residuals, alpha=0.5)
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
plt.savefig("residual_analysis.png", dpi=100, bbox_inches='tight')
plt.show()
print("Residual analysis plot saved as 'residual_analysis.png'")

# 8. MODEL COMPARISON
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

print("\nCross-validation RMSE Scores:")
for model, score in sorted(model_performances.items(), key=lambda x: x[1]):
    print(f"{model:20s}: ${score:,.2f}")

# 9. SAVE THE FINAL MODEL AND PIPELINE
print("\n" + "="*60)
print("SAVING THE FINAL MODEL AND PIPELINE")
print("="*60)

import joblib

# Save the model and pipeline
joblib.dump(final_model, 'final_housing_model.pkl')
joblib.dump(full_pipeline, 'full_pipeline.pkl')
print("\nModel saved as 'final_housing_model.pkl'")
print("Pipeline saved as 'full_pipeline.pkl'")

# 10. EXAMPLE PREDICTION
print("\n" + "="*60)
print("EXAMPLE PREDICTION")
print("="*60)

# Get a sample from the test set
sample_data = X_test.iloc[:5]
sample_labels = y_test.iloc[:5]
sample_prepared = full_pipeline.transform(sample_data)
sample_predictions = final_model.predict(sample_prepared)

print("\nSample Predictions vs Actual Values:")
for i in range(5):
    print(f"Sample {i+1}: Predicted = ${sample_predictions[i]:,.2f}, "
          f"Actual = ${sample_labels.iloc[i]:,.2f}, "
          f"Error = ${abs(sample_predictions[i] - sample_labels.iloc[i]):,.2f}")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)