"""
Workflow
========
1. **Load** spectral data from `IDRCShootOut2010Completo.mat` (hemoglobin is
   the **first** target column).
2. **Evolve** a binary mask over the 700 wavelength channels of **transmittance**
   (calibration → validation) minimizing RMSE.
3. **Train** a Multiple Linear Regression (MLR) model on **reflectance** data
   restricted to the selected wavelengths.
4. **Evaluate** on the reflectance test set, reporting **R², RMSE, MAE, Bias, SEP**.
5. **Persist** selected indices/λ (nm) and metrics under `./outputs`.

Usage
-----
$ python ga.py --mat-file data/IDRCShootOut2010Completo.mat --out ./outputs
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pygad
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###############################################################################
# Data loading                                                                #
###############################################################################


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


###############################################################################
# Genetic Algorithm                                                           #
###############################################################################


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

    # ---------------------------------------------------------------------
    # Fitness function (PyGAD ≥ 2.20 ➜ 3 positional args)
    # ---------------------------------------------------------------------
    def fitness_func(ga_instance, solution, sol_idx):  # noqa: D401,E501
        mask = solution >= 0.5  # Continuous genes → binary mask
        if not np.any(mask):  # Penalise empty selection
            return -1e6
        model = LinearRegression()
        model.fit(Xtrain[:, mask], ytrain)
        pred = model.predict(Xval[:, mask])
        rmse = np.sqrt(mean_squared_error(yval, pred))
        return -rmse  # PyGAD maximises fitness

    # ---------------------------------------------------------------------
    # Callback every generation (single-arg signature)
    # ---------------------------------------------------------------------
    def on_generation(ga_instance):  # noqa: D401
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
    return selected_idx.tolist(), best_rmse


###############################################################################
# Evaluation                                                                  #
###############################################################################


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
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    mae = mean_absolute_error(ytest, pred)
    bias = float(np.mean(pred - ytest))
    sep = float(np.sqrt(np.mean((pred - ytest - bias) ** 2)))  # Bias‐corrected RMSE
    r2 = r2_score(ytest, pred)
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": bias,
        "SEP": sep,
    }


###############################################################################
# Entry-point                                                                 #
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="GA-based wavelength selection for hemoglobin prediction"
    )
    parser.add_argument(
        "--mat-file",
        default="IDRCShootOut2010Completo.mat",
        help="Path to .mat data file",
    )
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

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
    ) = load_data(args.mat_file)

    selected_idx, val_rmse = select_wavelengths_ga(Xcal_tr, ycal_tr, Xval_tr, yval_tr)

    # Persist indices & wavelengths
    np.savetxt(os.path.join(args.out, "selected_indices.txt"), selected_idx, fmt="%d")
    np.savetxt(
        os.path.join(args.out, "selected_wavelengths_nm.txt"),
        wl_axis[selected_idx],
        fmt="%.2f",
    )

    # Final test evaluation
    metrics = evaluate_model(Xcal_rf, ycal_rf, Xtest_rf, ytest_rf, selected_idx)
    with open(os.path.join(args.out, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    print("\n========= SUMMARY ===========")
    print(f"Validation RMSE (Transmittance) : {val_rmse:8.4f}")
    print(f"Number of wavelengths selected  : {len(selected_idx)}")
    for k, v in metrics.items():
        print(f"{k:>5s} : {v:8.4f}")


if __name__ == "__main__":
    main()
