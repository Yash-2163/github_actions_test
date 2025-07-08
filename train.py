import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load data 
df = pd.read_csv("Salary_Data.csv")

# Feature and target split
X = df[['YearsExperience']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with Decision Tree
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)
print("Test MSE:", mse)

# Save model to final_model/best_model.pkl
os.makedirs("final_model", exist_ok=True)
joblib.dump(best_model, "final_model/best_model.pkl")
print("Model saved to final_model/best_model.pkl")
