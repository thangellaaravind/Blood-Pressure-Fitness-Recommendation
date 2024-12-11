import pandas as pd
import numpy as np
import os
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap
import joblib

# Load Dataset (for training purposes)
data = pd.read_csv('C:/Users/Aravind/Downloads/SIHHACK/bp/updated_sports_bp_dataset1.csv')

# Define Features and Targets
features = [
    'Age', 'Weight', 'Height', 'Cholesterol', 'Exercise', 'Smoking',
    'Stress_Score', 'Alcohol_Units_Per_Week', 'BMI_Age_Interaction',
    'Stress_Smoking_Interaction', 'Alcohol_Exercise_Interaction', 'education',
    'currentSmoker', 'prevalentHyp', 'heartRate', 'Hypertension_Score',
    'Hypertension_BMI_Interaction'
]
X = data[features]
y_systolic = data['Systolic_BP']
y_diastolic = data['Diastolic_BP']

# Split Data into Training and Test Sets
X_train, X_test, y_train_systolic, y_test_systolic = train_test_split(X, y_systolic, test_size=0.2, random_state=42)
_, X_test, y_train_diastolic, y_test_diastolic = train_test_split(X, y_diastolic, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create XGBRegressor Models for Systolic and Diastolic BP
systolic_model = XGBRegressor(objective='reg:squarederror')
diastolic_model = XGBRegressor(objective='reg:squarederror')

# Hyperparameter Grid for RandomizedSearchCV (adjusting for XGBoost parameters)
param_dist = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Hyperparameter Tuning for Systolic BP Model using RandomizedSearchCV
random_search_systolic = RandomizedSearchCV(
    systolic_model, param_distributions=param_dist, n_iter=50, cv=3,
    scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42
)
random_search_systolic.fit(X_train_scaled, y_train_systolic)
best_model_systolic = random_search_systolic.best_estimator_
joblib.dump(best_model_systolic, 'systolic_bp_model_xgb.pkl')

# Hyperparameter Tuning for Diastolic BP Model using RandomizedSearchCV
random_search_diastolic = RandomizedSearchCV(
    diastolic_model, param_distributions=param_dist, n_iter=50, cv=3,
    scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42
)
random_search_diastolic.fit(X_train_scaled, y_train_diastolic)
best_model_diastolic = random_search_diastolic.best_estimator_
joblib.dump(best_model_diastolic, 'diastolic_bp_model_xgb.pkl')

# Evaluate Models and Calculate AAMI and BHS Standards
def evaluate_and_calculate_standards(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errors = y_pred - y_test
    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)
    abs_errors = np.abs(errors)
    
    # AAMI Compliance
    aami_compliant = (abs(mean_error) <= 5) and (std_dev_error <= 8)
    
    # BHS Compliance
    within_5 = np.sum(abs_errors <= 5) / len(abs_errors) * 100
    within_10 = np.sum(abs_errors <= 10) / len(abs_errors) * 100
    within_15 = np.sum(abs_errors <= 15) / len(abs_errors) * 100
    
    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        bhs_grade = "A"
    elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        bhs_grade = "B"
    elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        bhs_grade = "C"
    else:
        bhs_grade = "D"
    
    # Print Results
    print(f"{label} Evaluation:")
    print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
    print(f"Mean Error (AAMI): {mean_error:.2f}, Standard Deviation (AAMI): {std_dev_error:.2f}")
    print(f"AAMI Compliance: {'Yes' if aami_compliant else 'No'}")
    print(f"BHS Grading: Within 5 mmHg: {within_5:.2f}%, Within 10 mmHg: {within_10:.2f}%, Within 15 mmHg: {within_15:.2f}%")
    print(f"BHS Grade: {bhs_grade}")
    print("-" * 50)
    
    return y_pred

# Evaluate Systolic BP Model
y_pred_systolic = evaluate_and_calculate_standards(best_model_systolic, X_test_scaled, y_test_systolic, "Systolic_BP")

# Evaluate Diastolic BP Model
y_pred_diastolic = evaluate_and_calculate_standards(best_model_diastolic, X_test_scaled, y_test_diastolic, "Diastolic_BP")

# SHAP Analysis for Systolic BP
explainer_systolic = shap.Explainer(best_model_systolic, X_train_scaled)
shap_values_systolic = explainer_systolic(X_test_scaled, check_additivity=False)

# SHAP Summary Plot
shap.summary_plot(shap_values_systolic, X_test_scaled, feature_names=features)

# SHAP Dependence Plot for Age
shap.dependence_plot('Age', shap_values_systolic.values, X_test_scaled, feature_names=features)

# Google Fit Integration for Real-Time Data
def authenticate_google_fit():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/fitness.activity.read'])
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Redirect to OAuth flow if needed (user has to authenticate)
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', ['https://www.googleapis.com/auth/fitness.activity.read'])
            creds = flow.run_local_server(port=0)

        # Save credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('fitness', 'v1', credentials=creds)
    return service

def get_google_fit_data(service):
    try:
        data_sources = service.users().dataSources().list(userId='me').execute()
        steps_data = None
        for data_source in data_sources.get('dataSource', []):
            if 'steps' in data_source['dataStreamId']:
                steps_data = service.users().dataSources().dataPointChanges().list(
                    userId='me', dataSourceId=data_source['dataStreamId']).execute()

        if not steps_data:
            print("No step data found from Google Fit.")
            return None

        return steps_data
    except Exception as e:
        print(f"Error fetching data from Google Fit: {str(e)}")
        return None

def predict_from_google_fit():
    service = authenticate_google_fit()
    steps_data = get_google_fit_data(service)

    if not steps_data:
        print("Unable to proceed without step data.")
        return

    # Extract steps from Google Fit data
    steps = 0
    for point in steps_data.get('point', []):
        steps += point['value'][0]['intVal']

    # Example static data (could be from user input in the app)
    input_data = {
        'Age': 30,  # Example static input or fetched from user profile
        'Weight': 70,
        'Height': 170,
        'Cholesterol': 180,
        'Exercise': 1,  # This could be calculated based on steps data
        'Smoking': 0,
        'Stress_Score': 5,
        'Alcohol_Units_Per_Week': 2,
        'BMI_Age_Interaction': 25,
        'Stress_Smoking_Interaction': 0,
        'Alcohol_Exercise_Interaction': 2,
        'education': 1,
        'currentSmoker': 0,
        'prevalentHyp': 0,
        'heartRate': 70,
        'Hypertension_Score': 5,
        'Hypertension_BMI_Interaction': 30
    }

    # Convert input data to dataframe
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[features])

    # Predict BP based on Google Fit data
    systolic_pred = best_model_systolic.predict(input_scaled)
    diastolic_pred = best_model_diastolic.predict(input_scaled)

    print(f"Predicted Systolic BP: {systolic_pred[0]:.2f}")
    print(f"Predicted Diastolic BP: {diastolic_pred[0]:.2f}")

# Call the prediction function with Google Fit data
predict_from_google_fit()
