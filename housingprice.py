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
    ("SVM Regressor",       SVR(kernel="linear")),
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

# 6. HYPERPARAMETER TUNING AND EXPERIMENTATION
print("\n" + "="*60)
print("HYPERPARAMETER TUNING AND EXPERIMENTATION")
print("="*60)

# 6.1 Try SVM Regressor (SVR) with RandomizedSearchCV
print("\nPerforming Randomized Search for SVR...")
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svr_reg = SVR()
rnd_search_svr = RandomizedSearchCV(svr_reg, param_distributions=param_distribs,
                                    n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                    verbose=2, random_state=42, n_jobs=-1)
rnd_search_svr.fit(housing_prepared, housing_labels)

svr_rmse = np.sqrt(-rnd_search_svr.best_score_)
print(f"\nBest SVR parameters found: {rnd_search_svr.best_params_}")
print(f"Best SVR cross-validation score (RMSE): ${svr_rmse:,.2f}")

# 6.2 Randomized Search for Random Forest (Replacing GridSearchCV)
print("\nPerforming Randomized Search for Random Forest...")
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search_forest = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                        random_state=42, n_jobs=-1)
rnd_search_forest.fit(housing_prepared, housing_labels)

forest_rmse = np.sqrt(-rnd_search_forest.best_score_)
print(f"\nBest Random Forest parameters found: {rnd_search_forest.best_params_}")
print(f"Best Random Forest cross-validation score (RMSE): ${forest_rmse:,.2f}")

# 7. CREATE A UNIFIED PIPELINE (PREPARATION + SELECTION + PREDICTION)
print("\n" + "="*60)
print("CREATING A UNIFIED PIPELINE")
print("="*60)

# Use Random Forest as the best model for the final pipeline
final_model = rnd_search_forest.best_estimator_

# Get feature names for importance sorting
feature_importances = final_model.feature_importances_
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + ['rooms_per_household', 'population_per_household', 
                            'bedrooms_per_room'] + cat_one_hot_attribs

k = 5 # Number of top features to select
full_pipeline_with_predictor = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances, k)),
    ("prediction", final_model)
])

# Train the unified pipeline
full_pipeline_with_predictor.fit(housing, housing_labels)
print("\nUnified pipeline (preparation + selection + prediction) trained successfully!")

# 8. AUTOMATICALLY EXPLORE PREPARATION OPTIONS
print("\n" + "="*60)
print("EXPLORING PREPARATION OPTIONS WITH GRIDSEARCH")
print("="*60)

param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(full_pipeline_with_predictor, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search_prep.fit(housing, housing_labels)

print(f"\nBest preparation options: {grid_search_prep.best_params_}")
print(f"Best score with tuned preparation: {np.sqrt(-grid_search_prep.best_score_):,.2f}")

# Update final model to the best one from prep search
final_pipeline = grid_search_prep.best_estimator_

# 9. EVALUATE ON TEST SET
print("\n" + "="*60)
print("EVALUATING FINAL MODEL ON TEST SET")
print("="*60)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Use the unified pipeline for prediction (prepares data automatically)
final_predictions = final_pipeline.predict(X_test)

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

# Save the unified final pipeline
joblib.dump(final_pipeline, 'final_housing_pipeline_complete.pkl')
print("\nFinal unified pipeline saved as 'final_housing_pipeline_complete.pkl'")

# 10. EXAMPLE PREDICTION
print("\n" + "="*60)
print("EXAMPLE PREDICTION")
print("="*60)

# Get a sample from the test set
sample_data = X_test.iloc[:5]
sample_labels = y_test.iloc[:5]
sample_predictions = final_pipeline.predict(sample_data)

print("\nSample Predictions vs Actual Values:")
for i in range(5):
    print(f"Sample {i+1}: Predicted = ${sample_predictions[i]:,.2f}, "
          f"Actual = ${sample_labels.iloc[i]:,.2f}, "
          f"Error = ${abs(sample_predictions[i] - sample_labels.iloc[i]):,.2f}")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)