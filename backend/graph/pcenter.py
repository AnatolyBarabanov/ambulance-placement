
"""
Problem:
Minimize the maximum ambulance travel time using
a limited number of candidate station locations.

Input:
    matrix: Candidate-to-neighborhood travel-time matrix.
    p: Number of ambulance stations to select.

Output:
    Selected station IDs, nearest-station travel times,
    maximum travel time and placement history.

Design:
    Start with the most central candidate.
    Repeatedly identify the worst-served neighborhood
    and select the unused candidate closest to it.

Time Complexity:
    O(k*n + p*(k+n)) with a precomputed matrix,
    where k is the number of candidates and n is
    the number of neighborhoods.

Space Complexity:
    O(k*n) including the input travel-time matrix.
    O(n+p) auxiliary space, excluding the matrix.

The algorithm is greedy and does not guarantee
a globally optimal station placement.
"""

import json
import math
import time

from backend.graph.traversal import build_adjacency
from backend.graph.dijkstra import travel_time_matrix


def greedy_pcenter(matrix, p):
    """Select p ambulance stations using greedy p-center."""

    candidates = list(matrix)

    if not candidates:
        raise ValueError("No candidate stations available.")

    if p < 1 or p > len(candidates):
        raise ValueError("Invalid number of stations.")

    neighborhoods = list(matrix[candidates[0]])

    if not neighborhoods:
        raise ValueError("No neighborhoods available.")

    # Step 1: Find the most central candidate
    first_station = min(
        candidates,
        key=lambda candidate: max(
            matrix[candidate].values()
        )
    )

    selected = [first_station]
    selected_set = {first_station}

    # Travel time to the nearest selected station
    nearest = dict(matrix[first_station])

    history = [{
        "stations": 1,
        "added_station": first_station,
        "maximum_time": max(nearest.values())
    }]

    # Continue until p stations have been selected
    while len(selected) < p:

        # Step 2: Identify worst-served neighborhood
        worst_neighborhood = max(
            neighborhoods,
            key=lambda node: nearest[node]
        )

        # Step 3: Identify unused candidate stations
        available = [
            candidate
            for candidate in candidates
            if candidate not in selected_set
        ]

        # Step 4: Find candidate closest to worst area
        next_station = min(
            available,
            key=lambda candidate: matrix[candidate][
                worst_neighborhood
            ]
        )

        selected.append(next_station)
        selected_set.add(next_station)

        # Step 5: Update nearest-station travel times
        for node in neighborhoods:

            nearest[node] = min(
                nearest[node],
                matrix[next_station][node]
            )

        history.append({
            "stations": len(selected),
            "added_station": next_station,
            "worst_neighborhood": worst_neighborhood,
            "maximum_time": max(nearest.values())
        })

    return {
        "selected_stations": selected,
        "nearest_times": nearest,
        "maximum_time": max(nearest.values()),
        "average_time": (
            sum(nearest.values()) / len(neighborhoods)
        ),
        "history": history
    }


def evaluate_coverage(county_map, nearest_times, target):
    """Calculate neighborhood and population coverage."""

    neighborhoods = county_map["neighborhoods"]

    total_population = sum(
        node["population"]
        for node in neighborhoods
    )

    covered_neighborhoods = 0
    covered_population = 0

    for node in neighborhoods:

        if nearest_times[node["id"]] <= target:

            covered_neighborhoods += 1
            covered_population += node["population"]

    return {
        "covered_neighborhoods": covered_neighborhoods,
        "neighborhood_coverage": (
            covered_neighborhoods
            / len(neighborhoods) * 100
        ),
        "covered_population": covered_population,
        "population_coverage": (
            covered_population
            / total_population * 100
        )
    }


if __name__ == "__main__":

    with open(
        "data/sample_map_500.json",
        "r",
        encoding="utf-8"
    ) as file:

        county_map = json.load(file)

    candidates = [
        node["id"]
        for node in county_map["neighborhoods"]
        if node["is_candidate"]
    ]

    normal_adj = build_adjacency(
        county_map,
        rush_hour=False
    )

    rush_adj = build_adjacency(
        county_map,
        rush_hour=True
    )

    normal_matrix = travel_time_matrix(
        normal_adj,
        candidates
    )

    rush_matrix = travel_time_matrix(
        rush_adj,
        candidates
    )

    station_budgets = [1, 3, 5, 10, 15, 20]

    target = 8

    print("GREEDY P-CENTER ANALYSIS")
    print("----------------------------")

    print("Neighborhoods:", len(normal_adj))
    print("Candidate stations:", len(candidates))
    print("Target travel time:", target, "minutes")

    for traffic, matrix in [
        ("NORMAL TRAFFIC", normal_matrix),
        ("RUSH-HOUR TRAFFIC", rush_matrix)
    ]:

        print("\n" + traffic)
        print("-" * 75)

        print(
            f"{'Stations':<10}"
            f"{'Average':<12}"
            f"{'Maximum':<12}"
            f"{'Pop. Cover':<14}"
            f"{'Node Cover':<14}"
            f"{'Time(ms)':<12}"
        )

        for p in station_budgets:

            start = time.perf_counter()

            result = greedy_pcenter(matrix, p)

            runtime = (
                time.perf_counter() - start
            ) * 1000

            coverage = evaluate_coverage(
                county_map,
                result["nearest_times"],
                target
            )

            print(
                f"{p:<10}"
                f"{result['average_time']:<12.2f}"
                f"{result['maximum_time']:<12.2f}"
                f"{coverage['population_coverage']:<14.2f}"
                f"{coverage['neighborhood_coverage']:<14.2f}"
                f"{runtime:<12.4f}"
            )

    # Detailed results for a 10-station placement
    result = greedy_pcenter(normal_matrix, 10)

    print("\nSELECTED STATIONS (NORMAL TRAFFIC)")
    print(result["selected_stations"])

    print("\nPLACEMENT HISTORY")
    print("----------------------------")

    for step in result["history"]:

        print(
            "Stations:", step["stations"],
            "| Added:", step["added_station"],
            "| Maximum travel time:",
            round(step["maximum_time"], 2)
        )

    # Verify that adding stations never increases
    # the maximum travel time
    history = result["history"]

    valid = all(
        history[i]["maximum_time"]
        <= history[i - 1]["maximum_time"]
        for i in range(1, len(history))
    )

    print("\nMaximum travel time verification:", valid)

    print(
        "Unique station verification:",
        len(set(result["selected_stations"])) == 10
    )

    print(
        "Candidate station verification:",
        all(
            station in candidates
            for station in result["selected_stations"]
        )
    )
