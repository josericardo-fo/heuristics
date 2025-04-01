import json
import random
import time
from math import exp

import numpy as np


def generate_board():
    """Gera um tabuleiro com um vetor, onde cada elemento representa a posição de uma rainha."""
    board = [random.randint(0, 7) for _ in range(8)]
    return board


def generate_neighbours(state):
    """Gera os vizinhos do estado atual, ou seja, todas as posições possíveis para a rainha."""
    neighbours = []
    for col in range(len(state)):
        for row in range(len(state)):
            if state[col] != row:
                neighbour = state.copy()
                neighbour[col] = row
                neighbours.append(neighbour)
    return neighbours


def count_conflicts(state):
    """Conta o número de conflitos entre as rainhas."""
    state_array = np.array(state)
    n = len(state_array)
    conflicts = 0

    # Verifica linhas (já é garantido que não há conflitos em colunas pela representação)
    rows_count = {}
    for row in state_array:
        rows_count[row] = rows_count.get(row, 0) + 1

    for count in rows_count.values():
        if count > 1:
            conflicts += (count * (count - 1)) // 2

    # Verifica diagonais
    for i in range(n):
        for j in range(i + 1, n):
            if abs(state_array[i] - state_array[j]) == abs(i - j):
                conflicts += 1

    return conflicts


def hill_climbing(state):
    """Algoritmo de Hill Climbing para resolver o problema das 8 rainhas."""
    iterations = 0
    while True:
        iterations += 1
        neighbours = generate_neighbours(state)
        current_conflicts = count_conflicts(state)
        best_neighbour = None
        best_conflicts = float("inf")

        for neighbour in neighbours:
            conflicts = count_conflicts(neighbour)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_neighbour = neighbour

        if best_conflicts >= current_conflicts:
            break
        state = best_neighbour

    return state, iterations


def simulated_annealing(
    state, initial_temp=250, cooling_rate=0.99, max_iterations=50000
):
    """Algoritmo de Simulated Annealing para resolver o problema das 8 rainhas."""
    current_state = state
    current_temp = initial_temp
    temperatures = [initial_temp]
    iterations = 0

    while current_temp > 0.01 and iterations < max_iterations:
        iterations += 1
        neighbours = generate_neighbours(current_state)
        next_state = random.choice(neighbours)

        current_conflicts = count_conflicts(current_state)
        next_conflicts = count_conflicts(next_state)

        delta = next_conflicts - current_conflicts
        if delta < 0:
            current_state = next_state
        else:
            # Aceita a mudança com uma probabilidade baseada na temperatura
            acceptance_probability = exp(-delta / current_temp)
            if random.random() < acceptance_probability:
                current_state = next_state

        # Atualiza a temperatura
        current_temp *= cooling_rate
        temperatures.append(current_temp)

        # Encerra se encontrou solução ótima
        if count_conflicts(current_state) == 0:
            break

    return current_state, temperatures, iterations


def run_experiments(num_experiments: int = 1000):
    """Executa os algoritmos n vezes e coleta dados de desempenho."""
    results = {
        "hill_climbing": {
            "success_rate": 0,
            "execution_times": [],
            "iterations": [],
            "conflicts": [],
        },
        "simulated_annealing": {
            "success_rate": 0,
            "execution_times": [],
            "iterations": [],
            "conflicts": [],
        },
    }

    for i in range(num_experiments):
        print(f"Experimento {i + 1}/{num_experiments}") if (i + 1) % 50 == 0 else ""
        initial_state = generate_board()

        # Hill Climbing
        start_time = time.time()
        hc_solution, hc_iterations = hill_climbing(initial_state)
        hc_time = time.time() - start_time
        hc_conflicts = count_conflicts(hc_solution)

        results["hill_climbing"]["execution_times"].append(hc_time)
        results["hill_climbing"]["iterations"].append(hc_iterations)
        results["hill_climbing"]["conflicts"].append(hc_conflicts)
        if hc_conflicts == 0:
            results["hill_climbing"]["success_rate"] += 1

        # Simulated Annealing
        start_time = time.time()
        sa_solution, _, sa_iterations = simulated_annealing(initial_state)
        sa_time = time.time() - start_time
        sa_conflicts = count_conflicts(sa_solution)

        results["simulated_annealing"]["execution_times"].append(sa_time)
        results["simulated_annealing"]["iterations"].append(sa_iterations)
        results["simulated_annealing"]["conflicts"].append(sa_conflicts)
        if sa_conflicts == 0:
            results["simulated_annealing"]["success_rate"] += 1

    # Calcular estatísticas
    summary_results = {"hill_climbing": {}, "simulated_annealing": {}}

    for algo in ["hill_climbing", "simulated_annealing"]:
        # Calcular as estatísticas para os dados brutos
        success_rate = (results[algo]["success_rate"] / num_experiments) * 100

        # Converter listas para arrays
        exec_times = np.array(results[algo]["execution_times"])
        iter_counts = np.array(results[algo]["iterations"])
        conflict_counts = np.array(results[algo]["conflicts"])

        avg_execution_time = np.mean(exec_times)
        avg_iterations = np.mean(iter_counts)
        avg_conflicts = np.mean(conflict_counts)
        min_conflicts = np.min(conflict_counts)
        max_conflicts = np.max(conflict_counts)

        # Atualizar resultados para o log no console
        results[algo]["success_rate"] = success_rate
        results[algo]["avg_execution_time"] = float(avg_execution_time)
        results[algo]["avg_iterations"] = float(avg_iterations)
        results[algo]["avg_conflicts"] = float(avg_conflicts)
        results[algo]["min_conflicts"] = int(min_conflicts)
        results[algo]["max_conflicts"] = int(max_conflicts)

        # Armazenar apenas dados resumidos para o JSON
        summary_results[algo] = {
            "success_rate": float(success_rate),
            "avg_execution_time": float(avg_execution_time),
            "avg_iterations": float(avg_iterations),
            "avg_conflicts": float(avg_conflicts),
            "min_conflicts": int(min_conflicts),
            "max_conflicts": int(max_conflicts),
            "num_experiments": num_experiments,
        }
        summary_results["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return results, summary_results


def save_results_to_json(summary_results, filename="eight_queens_results.json"):
    """Salva os resultados resumidos em um arquivo JSON."""
    filepath = f"./eight_queens/{filename}"
    with open(filepath, "w") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {filepath}.")


# Testando os algoritmos
if __name__ == "__main__":
    num_experiments = 1000
    results, summary_results = run_experiments(num_experiments)

    # Exibir resultados
    print("\nResultados dos experimentos:")
    for algo in ["hill_climbing", "simulated_annealing"]:
        print(f"\n{algo.replace('_', ' ').title()}:")
        print(f"Taxa de sucesso: {results[algo]['success_rate']:.2f}%")
        print(
            f"Tempo médio de execução: {results[algo]['avg_execution_time']:.6f} segundos"
        )
        print(f"Média de iterações: {results[algo]['avg_iterations']:.2f}")
        print(f"Média de conflitos restantes: {results[algo]['avg_conflicts']:.2f}")
        print(f"Mínimo de conflitos: {results[algo]['min_conflicts']}")
        print(f"Máximo de conflitos: {results[algo]['max_conflicts']}")

    # Salvar resultados em JSON
    save_results_to_json(summary_results)
