import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from mealpy import FloatVar, Problem, ACOR
from aco import RotatedEllipticProblem, ShiftedRotatedWeierstrassProblem

def run_experiments_with_tracking(num_runs=30):
    """
    Run multiple experiments and record convergence history
    
    Args:
        num_runs: Number of independent runs for statistical analysis
    
    Returns:
        dict: Dictionary containing results and convergence histories
    """
    n_dims = 10
    bounds_elliptic = FloatVar(lb=(-100.,) * n_dims, ub=(100.,) * n_dims, name="elliptic_vars")
    bounds_weierstrass = FloatVar(lb=(-0.5,) * n_dims, ub=(0.5,) * n_dims, name="weierstrass_vars")

    # Configurations to test
    configurations = [
        {"name": "Small", "pop_size": 20, "epoch": 500},
        {"name": "Medium", "pop_size": 50, "epoch": 1000},
        {"name": "Large", "pop_size": 100, "epoch": 2000}
    ]
    
    # Common parameters
    common_params = {
        "sample_count": 25,
        "intent_factor": 0.5,  # ρ: Pheromone evaporation rate
        "zeta": 1.0,  # α: Pheromone importance
        "verbose": False  # Turn off verbose output
    }
    
    # Store all results
    all_results = {
        "Rotated Elliptic": {config["name"]: {"fitness": [], "convergence": []} for config in configurations},
        "Shifted Rotated Weierstrass": {config["name"]: {"fitness": [], "convergence": []} for config in configurations}
    }
    
    # Run experiments
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        
        # Create new problem instances for each run to randomize rotation matrices
        elliptic_problem = RotatedEllipticProblem(bounds=bounds_elliptic)
        weierstrass_problem = ShiftedRotatedWeierstrassProblem(bounds=bounds_weierstrass)
        
        for config in configurations:
            print(f"\nConfig: {config['name']}")
            
            # Solve Rotated Elliptic
            model_elliptic = ACOR.OriginalACOR(
                epoch=config['epoch'], 
                pop_size=config['pop_size'], 
                **common_params
            )
            
            # Manually track convergence by modifying the algorithm's training process
            best_fitness_history = []
            
            def track_elliptic():
                # This function will be called after each epoch
                if hasattr(model_elliptic, 'history'):
                    if hasattr(model_elliptic.history, 'list_global_best_fit'):
                        best_fitness_history.append(model_elliptic.history.list_global_best_fit[-1])
                    else:
                        best_fitness_history.append(model_elliptic.history.list_global_best[-1][1])
                elif hasattr(model_elliptic, 'solution'):
                    best_fitness_history.append(model_elliptic.solution.target.fitness)
            
            # Solve the problem
            best_elliptic = model_elliptic.solve(elliptic_problem)
            
            # Extract convergence history if available
            if hasattr(model_elliptic, 'history') and hasattr(model_elliptic.history, 'list_global_best_fit'):
                convergence_data = model_elliptic.history.list_global_best_fit
            else:
                # Use the best fitness as a fallback
                convergence_data = [best_elliptic.target.fitness] * config['epoch']
            
            # Store results
            all_results["Rotated Elliptic"][config["name"]]["fitness"].append(best_elliptic.target.fitness)
            all_results["Rotated Elliptic"][config["name"]]["convergence"].append(convergence_data)
            
            # Solve Shifted Rotated Weierstrass
            model_weierstrass = ACOR.OriginalACOR(
                epoch=config['epoch'], 
                pop_size=config['pop_size'], 
                **common_params
            )
            
            # Solve the problem
            best_weierstrass = model_weierstrass.solve(weierstrass_problem)
            
            # Extract convergence history if available
            if hasattr(model_weierstrass, 'history') and hasattr(model_weierstrass.history, 'list_global_best_fit'):
                convergence_data = model_weierstrass.history.list_global_best_fit
            else:
                # Use the best fitness as a fallback
                convergence_data = [best_weierstrass.target.fitness] * config['epoch']
            
            # Store results
            all_results["Shifted Rotated Weierstrass"][config["name"]]["fitness"].append(best_weierstrass.target.fitness)
            all_results["Shifted Rotated Weierstrass"][config["name"]]["convergence"].append(convergence_data)
    
    # Fix convergence data by padding or truncating to match expected length
    for func_name in all_results:
        for config_name in all_results[func_name]:
            max_length = max(len(conv) for conv in all_results[func_name][config_name]["convergence"])
            
            for i, conv in enumerate(all_results[func_name][config_name]["convergence"]):
                if len(conv) < max_length:
                    # Pad with the last value
                    all_results[func_name][config_name]["convergence"][i] = list(conv) + [conv[-1]] * (max_length - len(conv))
                elif len(conv) > max_length:
                    # Truncate
                    all_results[func_name][config_name]["convergence"][i] = conv[:max_length]
    
    return all_results

def visualize_boxplots(results):
    """
    Create boxplots of final fitness values
    
    Args:
        results: Dictionary containing experiment results
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (func_name, func_results) in enumerate(results.items()):
        ax = axes[i]
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for config_name, config_results in func_results.items():
            data.append(config_results["fitness"])
            labels.append(config_name)
        
        # Create boxplot
        ax.boxplot(data, labels=labels)
        ax.set_title(f"{func_name} Function")
        ax.set_ylabel("Best Fitness Value (log scale)")
        ax.set_yscale('symlog')  # Use symlog to handle near-zero and negative values
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("aco_results_boxplot.png", dpi=300)
    plt.show()

def visualize_convergence(results):
    """
    Create convergence plots
    
    Args:
        results: Dictionary containing experiment results
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for func_name, func_results in results.items():
        plt.figure(figsize=(12, 6))
        
        for i, (config_name, config_results) in enumerate(func_results.items()):
            # Get convergence data
            convergence_data = np.array(config_results["convergence"])
            
            # For some configurations, we might have inconsistent data
            # Let's make sure all arrays are of the same length
            min_length = min(len(curve) for curve in convergence_data)
            convergence_data = np.array([curve[:min_length] for curve in convergence_data])
            
            # Calculate mean and std of convergence curves
            mean_convergence = np.mean(convergence_data, axis=0)
            std_convergence = np.std(convergence_data, axis=0)
            
            # Get x-axis (epochs)
            epochs = np.arange(1, len(mean_convergence) + 1)
            
            # Plot mean line
            plt.plot(epochs, mean_convergence, label=config_name, color=colors[i], linewidth=2)
            
            # Plot confidence interval
            plt.fill_between(epochs, 
                            mean_convergence - std_convergence, 
                            mean_convergence + std_convergence, 
                            alpha=0.2, color=colors[i])
        
        plt.title(f"{func_name} Function: Convergence Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Best Fitness (log scale)")
        plt.yscale('symlog')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"aco_convergence_{func_name.replace(' ', '_')}.png", dpi=300)
        plt.show()

def create_summary_table(results):
    """
    Create a summary table with statistics
    
    Args:
        results: Dictionary containing experiment results
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    
    for func_name, func_results in results.items():
        for config_name, config_results in func_results.items():
            fitness_values = config_results["fitness"]
            
            summary_data.append({
                "Function": func_name,
                "Configuration": config_name,
                "Mean": np.mean(fitness_values),
                "Std": np.std(fitness_values),
                "Min": np.min(fitness_values),
                "Max": np.max(fitness_values),
                "Median": np.median(fitness_values),
            })
    
    # Create DataFrame and sort by function and configuration
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by=["Function", "Configuration"])
    
    return summary_df

if __name__ == "__main__":
    # Run experiments (or load from saved file)
    try:
        print("Looking for saved results...")
        with open("aco_results.pkl", "rb") as f:
            all_results = pickle.load(f)
        print("Results loaded from file.")
    except FileNotFoundError:
        print("No saved results found. Running experiments...")
        all_results = run_experiments_with_tracking(num_runs=5)  # Using 5 runs for quicker demo
        # Save results for future use
        with open("aco_results.pkl", "wb") as f:
            pickle.dump(all_results, f)
    
    # Visualize results
    visualize_boxplots(all_results)
    visualize_convergence(all_results)
    
    # Create and display summary table
    summary = create_summary_table(all_results)
    print("\nSummary Statistics:")
    print(summary)
    
    # Save summary to CSV
    summary.to_csv("aco_results_summary.csv", index=False)
    print("\nSummary saved to 'aco_results_summary.csv'")