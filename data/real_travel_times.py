
import heapq
import math
import time
import numpy as np
import pandas as pd

from pathlib import Path

from real_county import (
    load_county_data,
    classify_neighborhoods,
    select_candidates
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Assumed speed for off-network access
ACCESS_SPEED_KMH = 30

# Simulated rush-hour congestion multipliers
TRAFFIC_FACTORS = {
    "motorway": 1.60,
    "trunk": 1.55,
    "primary": 1.50,
    "secondary": 1.40,
    "tertiary": 1.30,
    "residential": 1.15,
    "unclassified": 1.20,
    "living_street": 1.10
}

DEFAULT_TRAFFIC_FACTOR = 1.25


def build_adjacency_list(graph):

    print("\nBUILDING ROAD ADJACENCY LIST")
    print("=" * 50)

    adjacency = {
        node: [] for node in graph.nodes
    }

    for source, target, data in graph.edges(data=True):

        # OSMnx stores travel time in seconds
        normal_time = float(data["travel_time"]) / 60

        if not math.isfinite(normal_time) or normal_time < 0:
            raise ValueError("Invalid road travel time.")

        highway = data.get("highway", "unclassified")

        # Simplified roads can have multiple classifications
        if isinstance(highway, list):

            factor = max(
                TRAFFIC_FACTORS.get(
                    road_type,
                    DEFAULT_TRAFFIC_FACTOR
                )
                for road_type in highway
            )

        else:

            factor = TRAFFIC_FACTORS.get(
                highway,
                DEFAULT_TRAFFIC_FACTOR
            )

        rush_time = normal_time * factor

        adjacency[source].append(
            (target, normal_time, rush_time)
        )

    print("Vertices:", len(adjacency))

    print(
        "Directed edges:",
        sum(len(edges) for edges in adjacency.values())
    )

    return adjacency


def dijkstra(adjacency, source, rush_hour=False):

    distances = {
        node: math.inf for node in adjacency
    }

    distances[source] = 0.0

    priority_queue = [(0.0, source)]

    while priority_queue:

        current_distance, current_node = (
            heapq.heappop(priority_queue)
        )

        # Ignore outdated heap entries
        if current_distance > distances[current_node]:
            continue

        for neighbor, normal_time, rush_time in (
            adjacency[current_node]
        ):

            if rush_hour:
                weight = rush_time
            else:
                weight = normal_time

            new_distance = current_distance + weight

            # Relax the edge
            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    return distances


def calculate_access_time(distance_m):

    distance_km = distance_m / 1000

    time_minutes = (
        distance_km / ACCESS_SPEED_KMH
    ) * 60

    return time_minutes


def calculate_travel_matrix(
    adjacency,
    sources,
    neighborhoods,
    rush_hour=False
):

    traffic = "RUSH HOUR" if rush_hour else "NORMAL"

    print(f"\nCALCULATING {traffic} TRAVEL TIMES")
    print("=" * 50)

    start_time = time.perf_counter()

    matrix = np.full(
        (len(sources), len(neighborhoods)),
        np.inf
    )

    destination_nodes = (
        neighborhoods["road_node"].to_numpy()
    )

    destination_access = calculate_access_time(
        neighborhoods["snap_distance_m"].to_numpy()
    )

    # Reuse results if sources share a road node
    cache = {}

    for i, (_, station) in enumerate(sources.iterrows()):

        source_node = station["road_node"]

        if source_node not in cache:

            distances = dijkstra(
                adjacency,
                source_node,
                rush_hour=rush_hour
            )

            cache[source_node] = np.array([
                distances[node]
                for node in destination_nodes
            ])

        road_times = cache[source_node]

        source_access = calculate_access_time(
            station["snap_distance_m"]
        )

        matrix[i] = (
            source_access
            + road_times
            + destination_access
        )

        # A candidate located at its own demand point
        # has zero modeled travel distance to that point
        if "id" in sources.columns:

            same_location = (
                neighborhoods["id"].to_numpy()
                == station["id"]
            )

            matrix[i, same_location] = 0.0

        if (i + 1) % 10 == 0 or i + 1 == len(sources):

            print(
                f"Completed {i + 1}/{len(sources)} sources"
            )

    elapsed = time.perf_counter() - start_time

    if not np.isfinite(matrix).all():
        raise ValueError("Unreachable destinations detected.")

    print("\nMATRIX SUMMARY")
    print("-" * 40)

    print("Matrix shape:", matrix.shape)
    print("Travel time values:", matrix.size)

    print(
        "Minimum travel time:",
        round(matrix.min(), 2),
        "minutes"
    )

    print(
        "Maximum travel time:",
        round(matrix.max(), 2),
        "minutes"
    )

    print(
        "Average travel time:",
        round(matrix.mean(), 2),
        "minutes"
    )

    print("Unique Dijkstra executions:", len(cache))

    print(
        "Execution time:",
        round(elapsed, 2),
        "seconds"
    )

    return matrix

def save_matrix(matrix, name, source_ids, destination_ids):

    np.save(
        OUTPUT_DIR / f"{name}.npy",
        matrix
    )

    df = pd.DataFrame(
        matrix,
        index=source_ids,
        columns=destination_ids
    )

    df.index.name = "source_id"

    df.to_csv(
        OUTPUT_DIR / f"{name}.csv"
    )

    print("Saved:", name)

def main():

    start = time.perf_counter()

    # Load and prepare real county data
    neighborhoods, stations, graph = load_county_data()

    neighborhoods = classify_neighborhoods(
        neighborhoods
    )

    neighborhoods, candidates = select_candidates(
        neighborhoods
    )

    adjacency = build_adjacency_list(graph)

    # Existing EMS stations
    existing_normal = calculate_travel_matrix(
        adjacency,
        stations,
        neighborhoods,
        rush_hour=False
    )

    existing_rush = calculate_travel_matrix(
        adjacency,
        stations,
        neighborhoods,
        rush_hour=True
    )

    # Candidate station locations
    candidate_normal = calculate_travel_matrix(
        adjacency,
        candidates,
        neighborhoods,
        rush_hour=False
    )

    candidate_rush = calculate_travel_matrix(
        adjacency,
        candidates,
        neighborhoods,
        rush_hour=True
    )

    # Verification
    assert existing_normal.shape == (
        len(stations), len(neighborhoods)
    )

    assert candidate_normal.shape == (
        len(candidates), len(neighborhoods)
    )

    assert np.all(
        existing_rush >= existing_normal - 1e-9
    )

    assert np.all(
        candidate_rush >= candidate_normal - 1e-9
    )

    # Save all four matrices
    save_matrix(
        existing_normal,
        "existing_normal",
        stations["station_id"],
        neighborhoods["id"]
    )

    save_matrix(
        existing_rush,
        "existing_rush",
        stations["station_id"],
        neighborhoods["id"]
    )

    save_matrix(
        candidate_normal,
        "candidate_normal",
        candidates["id"],
        neighborhoods["id"]
    )

    save_matrix(
        candidate_rush,
        "candidate_rush",
        candidates["id"],
        neighborhoods["id"]
    )

    # Save metadata for future optimization and UI
    neighborhoods.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_DIR / "neighborhoods.csv",
        index=False
    )

    stations.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_DIR / "existing_stations.csv",
        index=False
    )

    candidates.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_DIR / "candidate_stations.csv",
        index=False
    )

    print("\nFINAL VERIFICATION")
    print("=" * 50)

    print("Existing normal:", existing_normal.shape)
    print("Existing rush:", existing_rush.shape)
    print("Candidate normal:", candidate_normal.shape)
    print("Candidate rush:", candidate_rush.shape)

    total_values = (
        existing_normal.size
        + existing_rush.size
        + candidate_normal.size
        + candidate_rush.size
    )

    print("Total travel time values:", total_values)
    print("Unreachable destinations: 0")
    print("Traffic consistency checks: Passed")

    print(
        "Total execution time:",
        round(time.perf_counter() - start, 2),
        "seconds"
    )

    print("\nTravel time generation successful.")


if __name__ == "__main__":

    main()
