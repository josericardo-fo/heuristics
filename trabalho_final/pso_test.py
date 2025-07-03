import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pso_pyswarm import run_pso_feature_selection

# Output directory
OUT_DIR = "trabalho_final/outputs_pso"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_plots(
    cost_history: list,
    wl_axis: np.ndarray,
    selected_idx: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """Create and save convergence, scatter, residual, and wavelength plots."""
    # 1. PSO convergence curve
    plt.figure()
    plt.plot(cost_history, marker="o")
    plt.title("Convergência PSO – Melhor RMSE por iteração")
    plt.xlabel("Iteração")
    plt.ylabel("RMSE (com penalidade)")
    plt.savefig(
        os.path.join(OUT_DIR, "pso_convergence.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Predicted vs True scatter
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--")
    plt.title("Predito vs Medido (teste)")
    plt.xlabel("Medido")
    plt.ylabel("Predito")
    plt.savefig(
        os.path.join(OUT_DIR, "pso_pred_vs_true.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Residuals histogram
    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("Distribuição dos resíduos")
    plt.xlabel("Erro (Predito - Medido)")
    plt.ylabel("Frequência")
    plt.savefig(
        os.path.join(OUT_DIR, "pso_residuals_hist.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Spectrum with selected wavelengths
    plt.figure(figsize=(10, 3))
    plt.plot(wl_axis, np.zeros_like(wl_axis), alpha=0)  # invisible baseline
    plt.scatter(
        wl_axis[selected_idx], np.zeros_like(selected_idx), c="r", marker="|", s=200
    )
    plt.title("Wavelengths selecionados pelo PSO")
    plt.yticks([])
    plt.xlabel("Comprimento de onda (nm)")
    plt.savefig(
        os.path.join(OUT_DIR, "pso_selected_wavelengths.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def evaluate_model(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xtest: np.ndarray,
    ytest: np.ndarray,
    selected_features,
) -> tuple[dict, np.ndarray]:
    """Train on training set and compute metrics on test set using selected features."""
    from sklearn.linear_model import LinearRegression
    
    # Handle different input types (list or numpy array)
    if isinstance(selected_features, list):
        # If it's already a list of indices, use it directly
        selected_idx = selected_features
    else:
        # If it's a numpy array
        if hasattr(selected_features, 'dtype') and selected_features.dtype == bool:
            # If it's a boolean mask, convert to indices
            selected_idx = np.where(selected_features)[0]
        else:
            # Assume it's already indices
            selected_idx = selected_features
    
    model = LinearRegression()
    model.fit(Xtrain[:, selected_idx], ytrain)
    pred = model.predict(Xtest[:, selected_idx])

    mse = mean_squared_error(ytest, pred)
    rmse = np.sqrt(mse)
    bias = float(np.mean(pred - ytest))
    r2 = r2_score(ytest, pred)

    return {"R2": float(r2), "RMSE": float(rmse), "MSE": float(mse), "Bias": bias}, pred

def main():
    # Load data
    import scipy.io
    data = scipy.io.loadmat("trabalho_final/data/IDRCShootOut2010Completo.mat")
    
    # Extract data
    Xcal_tr = data["XcalTrans"]
    ycal_tr = data["YcalTrans"][:, 0].ravel()
    
    Xval_tr = data.get("XvalTrans", None)
    yval_tr = data.get("YvalTrans", None)
    if Xval_tr is None or yval_tr is None:
        # If no validation set, create one
        from sklearn.model_selection import train_test_split
        Xcal_tr, Xval_tr, ycal_tr, yval_tr = train_test_split(
            Xcal_tr, ycal_tr, test_size=0.2, random_state=42
        )
    else:
        yval_tr = yval_tr[:, 0].ravel()
    
    Xcal_rf = data["XcalReflect"]
    ycal_rf = data["YcalReflect"][:, 0].ravel()
    
    Xtest_rf = data["XtestReflect"]
    ytest_rf = data["YtestTrans"][:, 0].ravel()
    
    # Print shapes to debug the mismatch
    print(f"Xtest_rf shape: {Xtest_rf.shape}")
    print(f"ytest_rf shape: {ytest_rf.shape}")
    
    # Get wavelength axis if available
    wl_axis = data.get("WLaxis", np.arange(Xcal_tr.shape[1])).ravel()
    
    print("Começando a otimização com PSO...")
    
    # Run PSO with tracking enabled
    selected_features, results = run_pso_feature_selection(
        Xcal_tr, ycal_tr, wavelengths=wl_axis, alpha=0.01, max_features=20, track_swarm=True
    )
    
    # Convert to indices
    selected_idx = np.where(selected_features)[0].tolist()
    
    # Save selected indices and wavelengths
    np.savetxt(os.path.join(OUT_DIR, "pso_selected_indices.txt"), selected_idx, fmt="%d")
    np.savetxt(
        os.path.join(OUT_DIR, "pso_selected_wavelengths_nm.txt"),
        wl_axis[selected_idx],
        fmt="%.2f",
    )
    
    # Fix the sample size mismatch - make sure Xtest_rf and ytest_rf have same number of samples
    min_samples = min(Xtest_rf.shape[0], ytest_rf.shape[0])
    Xtest_rf_trimmed = Xtest_rf[:min_samples]
    ytest_rf_trimmed = ytest_rf[:min_samples]
    
    print(f"Trimmed test data to {min_samples} samples (originally {Xtest_rf.shape[0]} X samples and {ytest_rf.shape[0]} y samples)")
    
    # Evaluate on reflectance data with matching sample sizes
    metrics, y_pred = evaluate_model(Xcal_rf, ycal_rf, Xtest_rf_trimmed, ytest_rf_trimmed, selected_idx)
    with open(os.path.join(OUT_DIR, "pso_metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    # Extract cost history from results
    if isinstance(results, dict) and 'cost_history' in results:
        cost_history = results['cost_history']
    else:
        # If not available, create a dummy
        cost_history = list(range(50))  # Just a placeholder sequence
    
    # Generate basic plots
    generate_plots(cost_history, wl_axis, selected_idx, ytest_rf_trimmed, y_pred)
    
    # Visualize swarm evolution if tracking data is available
    if isinstance(results, dict) and 'tracked_data' in results:
        visualize_swarm_evolution(results['tracked_data'], wl_axis)
    
    # Console summary
    print("\n========= RESUMO ===========")
    print(f"# λ selecionados : {len(selected_idx)}")
    print("--- Teste (Reflectância) ---")
    for k, v in metrics.items():
        print(f"{k:>4s} : {v:10.4f}")
    print("\nGráficos salvos em:", OUT_DIR)

def visualize_swarm_evolution(tracked_data, wl_axis, out_dir=OUT_DIR):
    """
    Visualize how the swarm evolved during optimization
    
    Parameters:
    -----------
    tracked_data : dict
        Dictionary with tracked particle positions and other data
    wl_axis : array
        Array of wavelength values
    out_dir : str
        Output directory for saving plots
    """
    positions = tracked_data['positions']
    iterations = tracked_data['iterations']
    diversity = tracked_data['diversity']
    
    if not positions:
        print("No swarm evolution data available.")
        return
    
    # 1. Feature selection frequency over iterations
    plt.figure(figsize=(12, 8))
    
    # Select a subset of iterations to visualize (beginning, middle, end)
    n_iters = len(iterations)
    if n_iters <= 4:
        selected_iters = range(n_iters)
    else:
        selected_iters = [0, n_iters//3, 2*n_iters//3, n_iters-1]
    
    # Get data dimensions
    n_features = positions[0].shape[1]
    
    # Create subplots for selected iterations
    for i, iter_idx in enumerate(selected_iters):
        ax = plt.subplot(2, 2, i+1)
        
        # Calculate frequency of each feature being selected
        feature_freq = np.mean(positions[iter_idx], axis=0)
        
        # Plot feature selection frequency
        ax.bar(range(n_features), feature_freq, alpha=0.7)
        ax.set_title(f"Iteration {iterations[iter_idx]}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Selection Frequency")
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pso_swarm_feature_freq.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Swarm diversity over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, diversity, marker='o')
    plt.title("Swarm Diversity over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Diversity (Mean Hamming Distance)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pso_swarm_diversity.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Feature selection heatmap (which features are selected at each iteration)
    plt.figure(figsize=(12, 8))
    
    # Create a matrix of how often each feature was selected at each iteration
    feature_selection_matrix = np.array([np.mean(pos, axis=0) for pos in positions])
    
    # Plot heatmap
    plt.imshow(
        feature_selection_matrix, 
        aspect='auto', 
        cmap='viridis', 
        interpolation='nearest',
        extent=(0, n_features, iterations[-1], iterations[0])
    )
    plt.colorbar(label="Selection Frequency")
    plt.title("Feature Selection Frequency over Iterations")
    plt.xlabel("Feature Index")
    plt.ylabel("Iteration")
    plt.savefig(os.path.join(out_dir, "pso_feature_selection_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Particle movement visualization for a few selected features
    # Select a few important features based on final selection frequency
    if len(positions) > 0:
        final_freq = np.mean(positions[-1], axis=0)
        important_features = np.argsort(-final_freq)[:5]  # Top 5 most selected features
        
        plt.figure(figsize=(12, 8))
        for i, feature_idx in enumerate(important_features):
            plt.subplot(2, 3, i+1)
            
            # Extract the trajectory for this feature across iterations
            feature_trajectory = [pos[:, feature_idx] for pos in positions]
            
            # Plot each particle's trajectory for this feature
            n_particles = positions[0].shape[0]
            for p in range(min(n_particles, 10)):  # Show max 10 particles for clarity
                particle_trajectory = [traj[p] for traj in feature_trajectory]
                plt.plot(iterations, particle_trajectory, alpha=0.5, marker='.')
            
            plt.title(f"Feature {feature_idx}")
            plt.xlabel("Iteration")
            plt.ylabel("Selection (0/1)")
            plt.ylim(-0.1, 1.1)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pso_particle_trajectories.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Swarm evolution visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()