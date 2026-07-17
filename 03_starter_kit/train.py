import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load environment variables
print("[INFO] Loading environment variables...")
load_dotenv(find_dotenv(), override=True)
# Fallback to script location's parent if PROJECT_ROOT is not in .env
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
print(f"[INFO] Project Root resolved to: {PROJECT_ROOT}")

# Generate Synthetic Airline Booking Dataset
def generate_flight_data(n_samples=5000):
    print(f"[PROCESS] Generating synthetic dataset ({n_samples} flight records)...")
    np.random.seed(42)
    
    days_to_departure = np.random.randint(1, 90, n_samples)
    total_capacity = np.random.choice([150, 180, 220, 300], n_samples)
    
    # Booked seats usually increase as days_to_departure decreases
    booking_progress = 1 - (days_to_departure / 90)
    booked_seats = (total_capacity * booking_progress * np.random.uniform(0.6, 1.0, n_samples)).astype(int)
    
    competitor_price = np.random.uniform(150, 500, n_samples)
    seasonality_index = np.random.uniform(0.8, 1.5, n_samples) # 1.0 is neutral
    
    # Optimal Fare calculation logic
    base_fare = 100
    scarcity_premium = (booked_seats / total_capacity) * 150
    urgency_premium = (90 - days_to_departure) * 1.5
    
    optimal_fare = (base_fare + scarcity_premium + urgency_premium) * seasonality_index
    # Adjust slightly towards competitor price
    optimal_fare = 0.7 * optimal_fare + 0.3 * competitor_price
    # Add some noise
    optimal_fare += np.random.normal(0, 10, n_samples)
    
    df = pd.DataFrame({
        'FlightId': [f'FL-{i:05d}' for i in range(n_samples)],
        'DaysToDeparture': days_to_departure,
        'BookedSeats': booked_seats,
        'TotalCapacity': total_capacity,
        'CompetitorPrice': competitor_price,
        'SeasonalityIndex': seasonality_index,
        'OptimalFare': np.maximum(50, optimal_fare) # Minimum fare is 50
    })
    print("[SUCCESS] Synthetic dataset generated successfully.")
    return df

def main():
    df = generate_flight_data(5000)
    dataset_dir = os.path.join(PROJECT_ROOT, "01_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, "flight_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Flight dataset saved locally at: {csv_path}")

    # Prepare features and target
    print("[PROCESS] Splitting dataset into training (80%) and validation (20%) sets...")
    features = ['DaysToDeparture', 'BookedSeats', 'TotalCapacity', 'CompetitorPrice', 'SeasonalityIndex']
    X = df[features]
    y = df['OptimalFare']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[PROCESS] Building pipeline and initializing GridSearchCV...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    # GridSearchCV for hyperparameter tuning
    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5]
    }
    
    print(f"[PROCESS] Grid searching over parameters: {param_grid}")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    
    print("[PROCESS] Training model (fitting 36 estimators)... Please wait.")
    grid_search.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    best_model = grid_search.best_estimator_

    # Evaluate
    print("[PROCESS] Evaluating model performance on validation split...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*40)
    print(f"|  MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"| Best Params: {grid_search.best_params_}")
    print(f"| Mean Squared Error (MSE): {mse:.4f}")
    print(f"| Mean Absolute Error (MAE): {mae:.4f}")
    print(f"| R2 Score (Accuracy): {r2:.4f}")
    print("="*40 + "\n")

    # Save Model for local inference
    output_dir = os.path.join(PROJECT_ROOT, "03_starter_kit")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    print(f"[PROCESS] Serializing and saving best estimator to: {model_path}")
    joblib.dump(best_model, model_path)
    print("[SUCCESS] Model saved successfully. Ready for deployment!")

if __name__ == "__main__":
    main()
