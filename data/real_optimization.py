import numpy as np
import pandas as pd
import time

from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/optimization")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_BUDGET = 40
COVERAGE_THRESHOLD = 8


def load_data():

    neighborhoods = pd.read_csv(
        DATA_DIR / "neighborhoods.csv"
    )

    candidates = pd.read_csv(
        DATA_DIR / "candidate_stations.csv"
    )

    stations = pd.read_csv(
        DATA_DIR / "existing_stations.csv"
    )

    matrices = {}

    for name in [
        "existing_normal",
        "existing_rush",
        "candidate_normal",
        "candidate_rush"
    ]:

        matrices[name] = np.load(
            DATA_DIR / f"{name}.npy"
        )

        # Verify matrix IDs against saved metadata
        csv_matrix = pd.read_csv(
            DATA_DIR / f"{name}.csv",
            index_col=0
        )

        sources = (
            stations["station_id"]
            if name.startswith("existing")
            else candidates["id"]
        )

        assert csv_matrix.index.astype(int).tolist() == (
            sources.astype(int).tolist()
        )

        assert csv_matrix.columns.astype(int).tolist() == (
            neighborhoods["id"].astype(int).tolist()
        )

        assert np.allclose(
            matrices[name],
            csv_matrix.to_numpy()
        )

        assert np.isfinite(matrices[name]).all()

    print("REAL COUNTY DATA LOADED")
    print("=" * 50)

    print("Neighborhoods:", len(neighborhoods))
    print("Candidates:", len(candidates))
    print("Existing EMS stations:", len(stations))
    print("Population:", int(neighborhoods["population"].sum()))

    return neighborhoods, candidates, stations, matrices


def greedy_p_center(matrix, budget):

    if budget < 1 or budget > matrix.shape[0]:
        raise ValueError("Invalid station budget.")

    start = time.perf_counter()

    num_candidates, num_neighborhoods = matrix.shape

    selected = []

    remaining = list(range(num_candidates))

    nearest_times = np.full(
        num_neighborhoods,
        np.inf
    )

    history = []

    for iteration in range(budget):

        best_candidate = None

        best_score = (np.inf, np.inf)

        best_times = None

        for candidate in remaining:

            # Calculate travel time if this station is added
            updated_times = np.minimum(
                nearest_times,
                matrix[candidate]
            )

            maximum_time = updated_times.max()

            average_time = updated_times.mean()

            # Primary objective: minimize maximum time
            # Tie-breaker: minimize average time
            score = (maximum_time, average_time)

            if score < best_score:

                best_candidate = candidate

                best_score = score

                best_times = updated_times

        selected.append(best_candidate)

        remaining.remove(best_candidate)

        nearest_times = best_times

        history.append({
            "iteration": iteration + 1,
            "candidate_row": best_candidate,
            "maximum_time": nearest_times.max(),
            "average_time": nearest_times.mean()
        })

        print(
            f"Station {iteration + 1}/{budget} | "
            f"Maximum: {nearest_times.max():.2f} min | "
            f"Average: {nearest_times.mean():.2f} min"
        )

    elapsed = time.perf_counter() - start

    print(
        "Optimization execution time:",
        round(elapsed, 4),
        "seconds"
    )

    return selected, nearest_times, pd.DataFrame(history)

def evaluate_placement(
    nearest_times,
    neighborhoods,
    threshold=8
):

    population = neighborhoods[
        "population"
    ].to_numpy()

    vulnerability = neighborhoods[
        "vulnerability"
    ].to_numpy()

    covered = nearest_times <= threshold

    vulnerability_weights = (
        population * vulnerability
    )

    metrics = {

        "average_time": nearest_times.mean(),

        "median_time": np.median(nearest_times),

        "maximum_time": nearest_times.max(),

        "minimum_time": nearest_times.min(),

        "95th_percentile": np.percentile(
            nearest_times, 95
        ),

        "population_weighted_time": np.average(
            nearest_times,
            weights=population
        ),

        "neighborhood_coverage": (
            covered.mean() * 100
        ),

        "population_coverage": (
            population[covered].sum()
            / population.sum() * 100
        ),

        "covered_population": int(
            population[covered].sum()
        ),

        "uncovered_population": int(
            population[~covered].sum()
        ),

        "vulnerability_weighted_time": np.average(
            nearest_times,
            weights=vulnerability_weights
        ),

        "vulnerability_weighted_coverage": (
            vulnerability_weights[covered].sum()
            / vulnerability_weights.sum() * 100
        )
    }

    return metrics

def compare_placements(
    neighborhoods,
    matrices,
    selected_normal,
    selected_rush
):

    placements = {

        "Existing EMS": {

            "normal": matrices[
                "existing_normal"
            ].min(axis=0),

            "rush": matrices[
                "existing_rush"
            ].min(axis=0)
        },

        "Normal Optimized": {

            "normal": matrices[
                "candidate_normal"
            ][selected_normal].min(axis=0),

            "rush": matrices[
                "candidate_rush"
            ][selected_normal].min(axis=0)
        },

        "Rush Optimized": {

            "normal": matrices[
                "candidate_normal"
            ][selected_rush].min(axis=0),

            "rush": matrices[
                "candidate_rush"
            ][selected_rush].min(axis=0)
        }
    }

    results = []

    for placement, scenarios in placements.items():

        for traffic, nearest_times in scenarios.items():

            metrics = evaluate_placement(
                nearest_times,
                neighborhoods,
                COVERAGE_THRESHOLD
            )

            results.append({
                "placement": placement,
                "traffic": traffic,
                **metrics
            })

    return pd.DataFrame(results), placements

def evaluate_area_types(neighborhoods, placements):

    results = []

    for placement, scenarios in placements.items():

        for traffic, nearest_times in scenarios.items():

            for area in ["urban", "suburban", "rural"]:

                mask = (
                    neighborhoods["area_type"]
                    .to_numpy() == area
                )

                if not mask.any():
                    continue

                area_data = neighborhoods.loc[mask]

                area_times = nearest_times[mask]

                metrics = evaluate_placement(
                    area_times,
                    area_data,
                    COVERAGE_THRESHOLD
                )

                results.append({

                    "placement": placement,

                    "traffic": traffic,

                    "area_type": area,

                    "neighborhoods": int(mask.sum()),

                    "population": int(
                        area_data["population"].sum()
                    ),

                    **metrics
                })

    return pd.DataFrame(results)

def main():

    start = time.perf_counter()

    neighborhoods, candidates, stations, matrices = (
        load_data()
    )

    print("\nNORMAL TRAFFIC OPTIMIZATION")
    print("=" * 50)

    selected_normal, normal_times, normal_history = (
        greedy_p_center(
            matrices["candidate_normal"],
            STATION_BUDGET
        )
    )

    print("\nRUSH HOUR OPTIMIZATION")
    print("=" * 50)

    selected_rush, rush_times, rush_history = (
        greedy_p_center(
            matrices["candidate_rush"],
            STATION_BUDGET
        )
    )

    # Verify optimization results
    assert len(selected_normal) == STATION_BUDGET

    assert len(selected_rush) == STATION_BUDGET

    assert len(set(selected_normal)) == STATION_BUDGET

    assert len(set(selected_rush)) == STATION_BUDGET

    assert np.all(
        np.diff(normal_history["maximum_time"]) <= 1e-9
    )

    assert np.all(
        np.diff(rush_history["maximum_time"]) <= 1e-9
    )

    # Retrieve selected station information
    normal_stations = candidates.iloc[
        selected_normal
    ].copy()

    rush_stations = candidates.iloc[
        selected_rush
    ].copy()

    normal_stations["selection_order"] = range(
        1, STATION_BUDGET + 1
    )

    rush_stations["selection_order"] = range(
        1, STATION_BUDGET + 1
    )

    # Compare placements
    results, placements = compare_placements(
        neighborhoods,
        matrices,
        selected_normal,
        selected_rush
    )

    area_results = evaluate_area_types(
        neighborhoods,
        placements
    )

    print("\nFINAL PLACEMENT COMPARISON")
    print("=" * 50)

    columns = [
        "placement",
        "traffic",
        "average_time",
        "maximum_time",
        "population_weighted_time",
        "population_coverage",
        "neighborhood_coverage"
    ]

    print(
        results[columns]
        .round(2)
        .to_string(index=False)
    )

    print("\nACCESSIBILITY BY AREA TYPE")
    print("=" * 50)

    print(
        area_results[
            [
                "placement",
                "traffic",
                "area_type",
                "average_time",
                "maximum_time",
                "population_coverage"
            ]
        ].round(2).to_string(index=False)
    )

    print("\nNORMAL OPTIMIZED STATION DISTRIBUTION")
    print("-" * 40)

    print(normal_stations["area_type"].value_counts())

    print("\nRUSH OPTIMIZED STATION DISTRIBUTION")
    print("-" * 40)

    print(rush_stations["area_type"].value_counts())

    overlap = len(
        set(selected_normal) & set(selected_rush)
    )

    print("\nSTATION PLACEMENT OVERLAP")
    print("-" * 40)

    print("Shared optimized stations:", overlap)

    print(
        "Overlap percentage:",
        round(overlap / STATION_BUDGET * 100, 2)
    )

    # Save optimization results
    results.to_csv(
        OUTPUT_DIR / "placement_comparison.csv",
        index=False
    )

    area_results.to_csv(
        OUTPUT_DIR / "area_type_comparison.csv",
        index=False
    )

    normal_stations.to_csv(
        OUTPUT_DIR / "optimized_normal_stations.csv",
        index=False
    )

    rush_stations.to_csv(
        OUTPUT_DIR / "optimized_rush_stations.csv",
        index=False
    )

    normal_history["candidate_id"] = (
        candidates.iloc[
            normal_history["candidate_row"].to_numpy()
        ]["id"].to_numpy()
    )

    rush_history["candidate_id"] = (
        candidates.iloc[
            rush_history["candidate_row"].to_numpy()
        ]["id"].to_numpy()
    )

    normal_history.to_csv(
        OUTPUT_DIR / "normal_optimization_history.csv",
        index=False
    )

    rush_history.to_csv(
        OUTPUT_DIR / "rush_optimization_history.csv",
        index=False
    )

    # Save individual neighborhood travel times
    neighborhood_results = neighborhoods.copy()

    for placement, scenarios in placements.items():

        for traffic, nearest_times in scenarios.items():

            name = placement.lower().replace(" ", "_")

            neighborhood_results[
                f"{name}_{traffic}_time"
            ] = nearest_times

    neighborhood_results.to_csv(
        OUTPUT_DIR / "neighborhood_accessibility.csv",
        index=False
    )

    print("\nFINAL VERIFICATION")
    print("=" * 50)

    print("Station budget:", STATION_BUDGET)

    print("Normal selected:", len(selected_normal))

    print("Rush selected:", len(selected_rush))

    print("Duplicate selections: 0")

    print("Optimization consistency checks: Passed")

    print(
        "Total execution time:",
        round(time.perf_counter() - start, 4),
        "seconds"
    )

    print("\nOptimization completed successfully.")


if __name__ == "__main__":

    main()
