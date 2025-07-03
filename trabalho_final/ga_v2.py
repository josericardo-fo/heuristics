import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygad
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

MAT_FILE = "trabalho_final/data/IDRCShootOut2010Completo.mat"
OUT_DIR = "trabalho_final/outputs_v2"


def load_data(mat_file: str) -> Tuple[np.ndarray, ...]:
    """Load all required splits from the .mat file and pick column 0 (hemoglobin)."""
    data = scipy.io.loadmat(mat_file)

    Xcal_trans = data["XcalTrans"]
    ycal_trans = data["YcalTrans"][:, 0].ravel()

    Xval_trans = data["XvalTrans"]
    yval_trans = data["YvalTrans"][:, 0].ravel()

    Xcal_ref = data["XcalReflect"]
    ycal_ref = data["YcalReflect"][:, 0].ravel()

    Xtest_ref = data["XtestReflect"]
    ytest_ref = data["YtestReflect"][:, 0].ravel()

    wl_axis = data.get("WLaxis", np.arange(Xcal_trans.shape[1])).ravel()
    return (
        Xcal_trans,
        ycal_trans,
        Xval_trans,
        yval_trans,
        Xcal_ref,
        ycal_ref,
        Xtest_ref,
        ytest_ref,
        wl_axis,
    )


def select_wavelengths_ga(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    n_generations: int = 50,
    pop_size: int = 80,
    mutation_percent_genes: int = 5,
    random_seed: int | None = 42,
) -> Tuple[List[int], float]:
    """Run GA to select feature subset. Returns indices and best validation RMSE."""

    n_features = Xtrain.shape[1]

    def fitness_func(ga_instance, solution, sol_idx):
        λ = 0.01
        mask = solution >= 0.5
        if not np.any(mask):
            return -1e6
        model = LinearRegression()
        model.fit(Xtrain[:, mask], ytrain)
        pred = model.predict(Xval[:, mask])
        rmse = np.sqrt(mean_squared_error(yval, pred))
        penalty = λ * mask.sum()
        return -(rmse + penalty)

    def on_generation(ga_instance):
        best_rmse = -ga_instance.best_solution()[1]
        print(
            f"Generation {ga_instance.generations_completed:3d} | "
            f"Val. RMSE: {best_rmse:8.4f}"
        )

    ga = pygad.GA(
        num_generations=n_generations,
        num_parents_mating=20,
        sol_per_pop=pop_size,
        num_genes=n_features,
        gene_type=float,
        init_range_low=0.0,
        init_range_high=1.0,
        mutation_percent_genes=mutation_percent_genes,
        crossover_type="single_point",
        mutation_type="random",
        parent_selection_type="sss",
        keep_parents=5,
        fitness_func=fitness_func,
        on_generation=on_generation,
        stop_criteria=["reach_0.0"],
        random_seed=random_seed,
    )

    ga.run()
    best_solution, best_fitness, _ = ga.best_solution()
    selected_idx = np.where(best_solution >= 0.5)[0]
    best_rmse = -best_fitness
    return selected_idx.tolist(), best_rmse, ga


def evaluate_model(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xtest: np.ndarray,
    ytest: np.ndarray,
    idx: List[int],
) -> Dict[str, float]:
    """Train on training set and compute metrics on test set using selected idx."""
    model = LinearRegression()
    model.fit(Xtrain[:, idx], ytrain)
    pred = model.predict(Xtest[:, idx])

    mse = mean_squared_error(ytest, pred)
    rmse = np.sqrt(mse)
    bias = float(np.mean(pred - ytest))
    r2 = r2_score(ytest, pred)

    return {"R2": float(r2), "RMSE": float(rmse), "MSE": float(mse), "Bias": bias}, pred


def generate_plots(
    ga: pygad.GA,
    wl_axis: np.ndarray,
    selected_idx: List[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """Create and save convergence, scatter, residual, and wavelength plots."""
    # 1. GA convergence curve
    plt.figure()
    plt.plot(-np.array(ga.best_solutions_fitness), marker="o")
    plt.title("Convergência GA – Melhor RMSE por geração")
    plt.xlabel("Geração")
    plt.ylabel("RMSE (validação)")
    plt.savefig(
        os.path.join(OUT_DIR, "ga_convergence.png"), dpi=300, bbox_inches="tight"
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
        os.path.join(OUT_DIR, "ga_pred_vs_true.png"), dpi=300, bbox_inches="tight"
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
        os.path.join(OUT_DIR, "ga_residuals_hist.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Spectrum with selected wavelengths
    plt.figure(figsize=(10, 3))
    plt.plot(wl_axis, np.zeros_like(wl_axis), alpha=0)  # invisible baseline
    plt.scatter(
        wl_axis[selected_idx], np.zeros_like(selected_idx), c="r", marker="|", s=200
    )
    plt.title("Wavelengths selecionados pelo GA")
    plt.yticks([])
    plt.xlabel("Comprimento de onda (nm)")
    plt.savefig(
        os.path.join(OUT_DIR, "ga_selected_wavelengths.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    (
        Xcal_tr,
        ycal_tr,
        Xval_tr,
        yval_tr,
        Xcal_rf,
        ycal_rf,
        Xtest_rf,
        ytest_rf,
        wl_axis,
    ) = load_data(MAT_FILE)

    print("Começando a otimização com Algoritmos Genéticos...")
    selected_idx, val_rmse, ga = select_wavelengths_ga(
        Xcal_tr, ycal_tr, Xval_tr, yval_tr
    )

    np.savetxt(os.path.join(OUT_DIR, "ga_selected_indices.txt"), selected_idx, fmt="%d")
    np.savetxt(
        os.path.join(OUT_DIR, "ga_selected_wavelengths_nm.txt"),
        wl_axis[selected_idx],
        fmt="%.2f",
    )

    metrics, y_pred = evaluate_model(Xcal_rf, ycal_rf, Xtest_rf, ytest_rf, selected_idx)
    with open(os.path.join(OUT_DIR, "ga_metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    generate_plots(ga, wl_axis, selected_idx, ytest_rf, y_pred)

    # Console summary
    print("\n========= RESUMO v2 ===========")
    print(f"RMSE (Val) : {val_rmse:8.4f}")
    print(f"# λ selecionados : {len(selected_idx)}")
    print("--- Teste (Reflectância) ---")
    for k, v in metrics.items():
        print(f"{k:>4s} : {v:10.4f}")
    print("\nGráficos salvos em:", OUT_DIR)


if __name__ == "__main__":
    main()
