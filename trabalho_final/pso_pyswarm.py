import pyswarms as ps

import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.plotters import plot_cost_history
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_feature_subset(selected_features, X, y, alpha=0.001, max_features=20):
    """
    Fitness function for PSO feature selection with MUCH gentler penalty
    
    Parameters:
    -----------
    selected_features : array
        Binary array where each row represents a particle's selected features
    X : array
        The feature matrix (spectral data)
    y : array
        The target values (hemoglobin)
    alpha : float
        Penalty weight for feature count (much lower for more features)
    max_features : int
        Target maximum number of features (now higher)
        
    Returns:
    --------
    fitness : array
        Fitness value for each particle (RMSE + penalty for too many features)
    """
    n_particles = selected_features.shape[0]
    fitness = np.zeros(n_particles)
    
    # For each particle (feature subset)
    for i in range(n_particles):
        # Get the selected features for this particle
        features = selected_features[i]
        
        # If no features are selected, return a high fitness value
        if np.sum(features) == 0:
            fitness[i] = 9999.0
            continue
            
        # Get the dataset with only the selected features
        X_subset = X[:, features.astype(bool)]
        
        # Evaluate using 5-fold cross-validation with MLR
        try:
            cv_scores = cross_val_score(
                LinearRegression(), 
                X_subset, 
                y, 
                cv=5, 
                scoring='neg_root_mean_squared_error'
            )
            # Convert negative RMSE to positive (PySwarms minimizes)
            rmse = -np.mean(cv_scores)
            
            # Get feature count
            n_features = np.sum(features)
            
            # VERY GENTLE PENALTY: Only penalize if far outside target range
            if n_features < 150:
                # Penalty for too few features
                penalty = alpha * (150 - n_features)**2
            elif n_features > 200:
                # Penalty for too many features
                penalty = alpha * (n_features - 200)**2
            else:
                # No penalty within our target range
                penalty = 0.0

            fitness[i] = rmse + penalty
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            fitness[i] = 9999.0
    
    return fitness

def binary_pso_feature_selection(X, y, n_particles=30, n_iterations=100, 
                                c1=0.5, c2=0.3, w=0.9, k=5, p=2):
    """
    Run Binary PSO for feature selection
    
    Parameters:
    -----------
    X : array
        The feature matrix (spectral data)
    y : array
        The target values (hemoglobin)
    n_particles : int
        Number of particles in the swarm
    n_iterations : int
        Number of iterations
    c1, c2, w : float
        PSO parameters (cognitive, social, inertia)
    k : int
        Number of neighbors for local topology
    p : int
        Norm order for distance calculations
        
    Returns:
    --------
    best_features : array
        Binary array of selected features
    best_score : float
        Fitness value of the best solution
    """
    # Define dimensions based on the number of features
    n_features = X.shape[1]
    
    # Setup options for PSO
    options = {
        'c1': c1,  # Cognitive parameter
        'c2': c2,  # Social parameter
        'w': w,    # Inertia parameter
        'k': k,    # Number of neighbors to consider
        'p': p     # Norm order
    }
    
    # Initialize Binary PSO
    optimizer = BinaryPSO(n_particles=n_particles, dimensions=n_features, options=options)
    
    # Define the objective function wrapper
    def obj_func(selected_features):
        return evaluate_feature_subset(selected_features, X, y)
    
    # Run the optimization
    print("Running Binary PSO for feature selection...")
    best_cost, best_pos = optimizer.optimize(obj_func, iters=n_iterations, verbose=True)
    
    # Get the selected features
    selected_features = best_pos.astype(bool)
    
    print(f"Optimization completed.")
    print(f"Best RMSE: {best_cost:.4f}")
    print(f"Number of selected features: {np.sum(selected_features)}/{n_features}")
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    plot_cost_history(optimizer.cost_history)
    plt.title("PSO Cost History")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (RMSE + penalty)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return selected_features, best_cost, optimizer.cost_history



def evaluate_selected_features(X, y, selected_features, test_size=0.3, random_state=42):
    """
    Evaluate the selected features with MLR
    
    Parameters:
    -----------
    X : array
        The feature matrix (spectral data)
    y : array
        The target values (hemoglobin)
    selected_features : array
        Binary array of selected features
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary of evaluation metrics
    """
    # Get the dataset with only the selected features
    X_subset = X[:, selected_features]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=test_size, random_state=random_state
    )
    
    # Train MLR model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    
    # Store results
    results = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'y_test': y_test,
        'y_pred': y_pred,
        'model': model
    }
    
    # Print results
    print("\nEvaluation of MLR with selected features:")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"Bias = {bias:.4f}")
    
    # Plot predictions vs. actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Hemoglobin")
    plt.ylabel("Predicted Hemoglobin")
    plt.title("MLR Performance with Selected Features")
    plt.grid(True)
    plt.annotate(
        f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    plt.tight_layout()
    plt.show()
    
    return results

def visualize_selected_wavelengths(selected_features, wavelengths=None):
    """
    Visualize which wavelengths were selected
    
    Parameters:
    -----------
    selected_features : array
        Binary array of selected features
    wavelengths : array, optional
        Array of wavelength values (if available)
    """
    indices = np.where(selected_features)[0]
    
    plt.figure(figsize=(12, 6))
    
    # If wavelengths are provided, use them for x-axis
    if wavelengths is not None and len(wavelengths) == len(selected_features):
        all_wavelengths = wavelengths
        selected_wavelengths = wavelengths[selected_features]
        
        # Plot all wavelengths as background
        plt.plot(all_wavelengths, np.zeros_like(all_wavelengths), 'k-', alpha=0.3)
        
        # Plot selected wavelengths as stems
        plt.stem(selected_wavelengths, np.ones_like(selected_wavelengths), 
                 linefmt='r-', markerfmt='ro', basefmt='r-')
        plt.xlabel("Wavelength (nm)")
    else:
        # Just use indices if wavelengths aren't available
        all_indices = np.arange(len(selected_features))
        
        # Plot all indices as background
        plt.plot(all_indices, np.zeros_like(all_indices), 'k-', alpha=0.3)
        
        # Plot selected indices as stems
        plt.stem(indices, np.ones_like(indices), 
                 linefmt='r-', markerfmt='ro', basefmt='r-')
        plt.xlabel("Feature Index")
    
    plt.ylabel("Selected (1) / Not Selected (0)")
    plt.title(f"Selected Features ({len(indices)} out of {len(selected_features)})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Selected feature indices: {indices}")
    if wavelengths is not None and len(wavelengths) == len(selected_features):
        print(f"Selected wavelengths: {wavelengths[selected_features]}")

def run_pso_feature_selection(X, y, wavelengths=None, alpha=0.01, max_features=20, track_swarm=False):
    """
    Complete pipeline for PSO feature selection
    
    Parameters:
    -----------
    X : array
        The feature matrix (spectral data)
    y : array
        The target values (hemoglobin)
    wavelengths : array, optional
        Array of wavelength values (if available)
    alpha : float
        Penalty weight for feature count
    max_features : int
        Target maximum number of features
    track_swarm : bool
        Whether to track swarm evolution for visualization
        
    Returns:
    --------
    selected_features : array
        Binary array of selected features
    results : dict
        Dictionary of evaluation metrics and history
    """
    # First key change: Use a HARD feature count limit in the objective function
    def custom_obj_func(selected_features):
        # For each particle, strictly limit to max_features
        limited_features = np.zeros_like(selected_features)
        
        for i in range(selected_features.shape[0]):
            # If more than max_features are selected, keep only the max_features highest ones
            if np.sum(selected_features[i]) > max_features:
                # Get indices sorted by feature values (decreasing)
                sorted_indices = np.argsort(-selected_features[i])
                # Keep only the top max_features
                limited_features[i, sorted_indices[:max_features]] = 1
            else:
                limited_features[i] = selected_features[i]
        
        # Evaluate with the strictly limited features
        return evaluate_feature_subset(limited_features, X, y, alpha=alpha)
    
    # Setup options for PSO
    n_features = X.shape[1]
    options = {
        'c1': 0.7,  # Increase cognitive parameter (personal learning)
        'c2': 0.5,  # Increase social parameter (global learning)
        'w': 0.8,   # Slightly reduce inertia for better convergence
        'k': 3,     # Fewer neighbors
        'p': 2      # Keep Euclidean distance
    }
    
    # Initialize Binary PSO
    optimizer = BinaryPSO(n_particles=50, dimensions=n_features, options=options)
    
    # Track swarm evolution if requested
    tracked_data = None
    tracked_wrapper = None
    if track_swarm:
        tracked_data, tracked_wrapper = track_swarm_evolution(optimizer, max_iters=50, save_every=2)
    
    # Run the optimization
    print(f"Running Binary PSO for feature selection (target features: 150-200)...")
    if track_swarm and tracked_wrapper is not None:
        # Use the tracking wrapper around the objective function
        wrapped_obj_func = tracked_wrapper(custom_obj_func)
        best_cost, best_pos = optimizer.optimize(wrapped_obj_func, iters=50, verbose=True)
    else:
        best_cost, best_pos = optimizer.optimize(custom_obj_func, iters=50, verbose=True)
    
    # Make sure features are within our desired range (150-200)
    best_pos_count = np.sum(best_pos)
    if best_pos_count < 150:
        # If too few features, add more until we reach at least 150
        missing = 150 - best_pos_count
        zero_indices = np.where(best_pos == 0)[0]
        add_indices = np.random.choice(zero_indices, min(missing, len(zero_indices)), replace=False)
        best_pos[add_indices] = 1
    elif best_pos_count > 200:
        # If too many features, limit to 200
        best_pos = limit_features(best_pos.reshape(1, -1), 200)[0]
    
    # Get the selected features
    selected_features = best_pos.astype(bool)
    
    print(f"Optimization completed.")
    print(f"Best RMSE: {best_cost:.4f}")
    print(f"Number of selected features: {np.sum(selected_features)}/{n_features}")
    
    # Visualize selected wavelengths
    visualize_selected_wavelengths(selected_features, wavelengths)
    
    # Evaluate with MLR
    results = evaluate_selected_features(X, y, selected_features)
    results['cost_history'] = optimizer.cost_history
    if tracked_data:
        results['tracked_data'] = tracked_data
    
    # Compare with full model (all features)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model_full = LinearRegression()
    model_full.fit(X_train_full, y_train_full)
    y_pred_full = model_full.predict(X_test_full)
    
    r2_full = r2_score(y_test_full, y_pred_full)
    rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
    
    print("\nComparison with Full Model:")
    print(f"Full Model - R²: {r2_full:.4f}, RMSE: {rmse_full:.4f}")
    print(f"Selected Features - R²: {results['r2']:.4f}, RMSE: {results['rmse']:.4f}")
    
    return selected_features, results

def limit_features(selected_features, max_features=200):
    """Force selection of at most max_features (now with higher limit)"""
    n_particles = selected_features.shape[0]
    limited_features = np.zeros_like(selected_features)
    
    for i in range(n_particles):
        # For each particle
        features = selected_features[i]
        n_selected = np.sum(features)
        
        if n_selected <= max_features:
            # Keep as is if under limit
            limited_features[i] = features
        else:
            # Get indices of selected features
            selected_indices = np.where(features == 1)[0]
            
            # Keep only the max_features most important ones
            # For this example, just randomly select max_features indices
            keep_indices = np.random.choice(selected_indices, max_features, replace=False)
            
            # Set only these indices to 1
            limited_features[i] = 0
            limited_features[i, keep_indices] = 1
            
    return limited_features
def track_swarm_evolution(optimizer, max_iters=50, save_every=5):
    """
    Tracks particle positions during PSO optimization
    
    Parameters:
    -----------
    optimizer : BinaryPSO
        The PSO optimizer
    max_iters : int
        Maximum number of iterations
    save_every : int
        Save particle positions every N iterations
        
    Returns:
    --------
    tracked_data, wrapper_function
    """
    # Storage for tracked data
    tracked_data = {
        'positions': [],      # Particle positions at each saved iteration
        'iterations': [],     # Iteration numbers that were saved
        'best_positions': [], # Global best position at each saved iteration
        'diversity': []       # Diversity measure at each saved iteration
    }
    
    # Counter for iterations
    iteration_count = [0]  # Using list for mutable reference
    
    # Create a wrapper for the objective function
    def wrapper(objective_func):
        def tracked_objective(position, **kwargs):
            # Call the original objective function
            costs = objective_func(position, **kwargs)
            
            # Save data at specified intervals
            if iteration_count[0] % save_every == 0 or iteration_count[0] == max_iters - 1:
                # Calculate diversity (mean pairwise Hamming distance)
                n_particles = position.shape[0]
                diversity = 0.0
                if n_particles > 1:
                    for i in range(n_particles):
                        for j in range(i+1, n_particles):
                            # Hamming distance between binary vectors
                            diversity += np.sum(position[i] != position[j])
                    diversity /= (n_particles * (n_particles - 1) / 2)
                
                # Store data
                tracked_data['positions'].append(position.copy())
                tracked_data['iterations'].append(iteration_count[0])
                if hasattr(optimizer, 'swarm') and hasattr(optimizer.swarm, 'best_pos'):
                    tracked_data['best_positions'].append(optimizer.swarm.best_pos.copy())
                tracked_data['diversity'].append(diversity)
                
                # Print update
                if iteration_count[0] % 10 == 0 or iteration_count[0] == max_iters - 1:
                    print(f"Iteration {iteration_count[0]}: Diversity = {diversity:.4f}")
            
            # Increment counter
            iteration_count[0] += 1
            return costs
        
        return tracked_objective
    
    return tracked_data, wrapper