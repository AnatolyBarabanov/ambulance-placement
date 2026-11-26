
"""
Problem:
Measure the execution time of graph and station
placement algorithms as the county size increases.

Input:
    County sizes, random seed, number of repetitions
    and station budget.

Output:
    CSV execution results and runtime comparison plots.

Design:
    Generate reproducible county maps, repeatedly
    execute each algorithm, collect runtime statistics
    and compare measured growth with theoretical
    asymptotic growth.
"""

import csv
import math
import time
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from backend.graph.generator import generate_map

from backend.graph.traversal import (
    build_adjacency,
    bfs_components,
    dfs_components
)

from backend.graph.dijkstra import (
    dijkstra,
    travel_time_matrix
)

from backend.graph.pcenter import greedy_pcenter


# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

MAP_SIZES = [50, 100, 200, 400, 800]

REPEATS = 30

SEED = 42

STATION_BUDGET = 10

OUTPUT_DIR = Path("experiments/results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# RUNTIME MEASUREMENT
# ==========================================

def measure_runtime(function, repeats=REPEATS):

    execution_times = []

    # Warm-up execution
    function()

    for _ in range(repeats):

        start = time.perf_counter()

        function()

        end = time.perf_counter()

        # Convert seconds to milliseconds
        elapsed = (end - start) * 1000

        execution_times.append(elapsed)

    return {
        "mean_ms": statistics.mean(execution_times),

        "median_ms": statistics.median(execution_times),

        "min_ms": min(execution_times),

        "max_ms": max(execution_times),

        "std_ms": statistics.stdev(execution_times)
    }


# ==========================================
# THEORETICAL TIME COMPLEXITIES
# ==========================================

def theoretical_growth(algorithm, v, e, k, p):

    if algorithm == "Map Generator":

        return v ** 2 * math.log2(v)

    elif algorithm in ["BFS", "DFS"]:

        return v + e

    elif algorithm == "Dijkstra":

        return (v + e) * math.log2(v)

    elif algorithm == "Travel-Time Matrix":

        return k * (v + e) * math.log2(v)

    elif algorithm == "Greedy P-Center":

        return k * v + p * (k + v)

    else:

        raise ValueError("Unknown algorithm.")


# ==========================================
# RUN EXPERIMENTS
# ==========================================

def run_experiments():

    results = []

    for n in MAP_SIZES:

        print("\n" + "=" * 60)
        print("COUNTY SIZE:", n)
        print("=" * 60)

        # Generate the county outside algorithm timing
        county = generate_map(n, SEED)

        adj = build_adjacency(
            county,
            rush_hour=False
        )

        candidates = [
            node["id"]
            for node in county["neighborhoods"]
            if node["is_candidate"]
        ]

        v = len(county["neighborhoods"])
        e = len(county["roads"])
        k = len(candidates)

        p = min(STATION_BUDGET, k)

        source = candidates[0]

        # Precompute matrix outside p-center timing
        matrix = travel_time_matrix(
            adj,
            candidates
        )

        print("Neighborhoods:", v)
        print("Roads:", e)
        print("Candidates:", k)
        print("Station budget:", p)

        algorithms = {
            "Map Generator": lambda: generate_map(
                n, SEED
            ),

            "BFS": lambda: bfs_components(adj),

            "DFS": lambda: dfs_components(adj),

            "Dijkstra": lambda: dijkstra(
                adj, source
            ),

            "Travel-Time Matrix": lambda: (
                travel_time_matrix(adj, candidates)
            ),

            "Greedy P-Center": lambda: (
                greedy_pcenter(matrix, p)
            )
        }

        for name, function in algorithms.items():

            stats = measure_runtime(function)

            growth = theoretical_growth(
                name, v, e, k, p
            )

            row = {
                "algorithm": name,
                "neighborhoods": v,
                "roads": e,
                "candidates": k,
                "stations": p,
                "repeats": REPEATS,
                **stats,
                "theoretical_growth": growth
            }

            results.append(row)

            print(
                f"{name:<22}"
                f"Median: {stats['median_ms']:.6f} ms"
            )

    return results


# ==========================================
# SAVE CSV RESULTS
# ==========================================

def save_results(results):

    filename = OUTPUT_DIR / "runtime_results.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV saved:", filename)


# ==========================================
# INDIVIDUAL RUNTIME PLOTS
# ==========================================

def plot_individual_algorithms(results):

    algorithms = list(dict.fromkeys(
        row["algorithm"]
        for row in results
    ))

    for algorithm in algorithms:

        rows = [
            row for row in results
            if row["algorithm"] == algorithm
        ]

        sizes = [
            row["neighborhoods"]
            for row in rows
        ]

        measured = [
            row["median_ms"]
            for row in rows
        ]

        theoretical = [
            row["theoretical_growth"]
            for row in rows
        ]

        # Scale theoretical growth to the first
        # measured execution time for comparison
        scale = measured[0] / theoretical[0]

        expected = [
            value * scale
            for value in theoretical
        ]

        plt.figure(figsize=(9, 6))

        plt.plot(
            sizes,
            measured,
            marker="o",
            linewidth=2,
            label="Measured median runtime"
        )

        plt.plot(
            sizes,
            expected,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Scaled theoretical growth"
        )

        plt.title(
            f"{algorithm}: Runtime Analysis"
        )

        plt.xlabel("Number of Neighborhoods")
        plt.ylabel("Execution Time (ms)")

        plt.xticks(MAP_SIZES)

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            algorithm.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        plt.savefig(
            OUTPUT_DIR / f"{filename}_runtime.png",
            dpi=300
        )

        plt.close()


# ==========================================
# COMBINED ALGORITHM COMPARISON
# ==========================================

def plot_combined_results(results):

    algorithms = list(dict.fromkeys(
        row["algorithm"]
        for row in results
    ))

    plt.figure(figsize=(11, 7))

    for algorithm in algorithms:

        rows = [
            row for row in results
            if row["algorithm"] == algorithm
        ]

        sizes = [
            row["neighborhoods"]
            for row in rows
        ]

        measured = [
            row["median_ms"]
            for row in rows
        ]

        plt.plot(
            sizes,
            measured,
            marker="o",
            linewidth=2,
            label=algorithm
        )

    # Logarithmic scale makes algorithms with
    # very different runtimes easier to compare
    plt.yscale("log")

    plt.title("Algorithm Runtime Comparison")

    plt.xlabel("Number of Neighborhoods")

    plt.ylabel(
        "Median Execution Time (ms, log scale)"
    )

    plt.xticks(MAP_SIZES)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = OUTPUT_DIR / "runtime_comparison.png"

    plt.savefig(filename, dpi=300)

    plt.close()

    print("Combined plot saved:", filename)


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("AMBULANCE PLACEMENT RUNTIME EXPERIMENTS")

    print("Map sizes:", MAP_SIZES)
    print("Repetitions:", REPEATS)
    print("Random seed:", SEED)
    print("Station budget:", STATION_BUDGET)

    start = time.perf_counter()

    results = run_experiments()

    save_results(results)

    plot_individual_algorithms(results)

    plot_combined_results(results)

    elapsed = time.perf_counter() - start

    print("\nEXPERIMENT SUMMARY")
    print("-" * 40)

    print("Algorithms tested:", 6)

    print("Map sizes tested:", len(MAP_SIZES))

    print("Result records:", len(results))

    print(
        "Total timed executions:",
        len(results) * REPEATS
    )

    print(
        "Total experiment duration:",
        round(elapsed, 2),
        "seconds"
    )

    print("Output folder:", OUTPUT_DIR)

    print("\nRuntime experiments completed.")
