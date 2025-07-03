from pso_pyswarm import run_pso_feature_selection
import numpy as np
import scipy.io

# Load the data from the .mat file
data = scipy.io.loadmat("data/IDRCShootOut2010Completo.mat")

# Extract the relevant data
X = data["XcalTrans"]  # Spectral data
y = data["YcalTrans"][:, 0]  # Hemoglobin values (first column)


# Run feature selection
selected_features, results = run_pso_feature_selection(X, y)

# Use selected features with reflectance data if needed
if "XcalReflect" in data and "XtestReflect" in data:
    # Extract reflectance data for training and testing
    X_train_reflect = data["XcalReflect"][:, selected_features]
    X_test_reflect = data["XtestReflect"][:, selected_features]
    
    # Make sure sample sizes match
    min_train_samples = min(X_train_reflect.shape[0], len(data["YcalTrans"][:, 0]))
    min_test_samples = min(X_test_reflect.shape[0], len(data["YtestTrans"][:, 0]))
    
    # Train MLR model on selected features with reflectance data
    from sklearn.linear_model import LinearRegression
    model_reflect = LinearRegression()
    model_reflect.fit(X_train_reflect[:min_train_samples], data["YcalTrans"][:min_train_samples, 0])
    
    # Make predictions
    y_pred_reflect = model_reflect.predict(X_test_reflect[:min_test_samples])
    
    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error
    r2_reflect = r2_score(data["YtestTrans"][:min_test_samples, 0], y_pred_reflect)
    rmse_reflect = np.sqrt(mean_squared_error(data["YtestTrans"][:min_test_samples, 0], y_pred_reflect))
    
    print(f"\nReflectance Model with Selected Features - R²: {r2_reflect:.4f}, RMSE: {rmse_reflect:.4f}")