
"""
Problem:
Find the shortest ambulance travel times from a starting
neighborhood to every other neighborhood in the road network.

Input:
    adj: Weighted adjacency list.
    source: Starting neighborhood ID.

Output:
    Dictionary containing the shortest travel time
    from the source to every neighborhood.

Design:
    Use Dijkstra's algorithm with a min-heap priority queue.
    Repeatedly process the neighborhood with the smallest
    known travel time and relax its neighboring edges.

Time Complexity:
    O((V + E) log V) for a simple graph.

Space Complexity:
    O(V + E), including the heap.

All road weights must be nonnegative.
"""

import heapq
import math
import json
import time

from backend.graph.traversal import build_adjacency


def dijkstra(adj, source):
    """Calculate shortest travel times from one source."""

    if source not in adj:
        raise ValueError("Source neighborhood does not exist.")

    # Initialize every neighborhood to infinity
    distances = {
        node: math.inf
        for node in adj
    }

    # Starting neighborhood has zero travel time
    distances[source] = 0

    # Min-heap stores (travel time, neighborhood ID)
    heap = [(0, source)]

    while heap:

        current_distance, current_node = heapq.heappop(heap)

        # Skip outdated heap entries
        if current_distance > distances[current_node]:
            continue

        # Examine neighboring neighborhoods
        for neighbor, weight in adj[current_node]:

            if weight < 0:
                raise ValueError(
                    "Dijkstra requires nonnegative weights."
                )

            new_distance = current_distance + weight

            # Relax the edge if a faster route is found
            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance

                heapq.heappush(
                    heap,
                    (new_distance, neighbor)
                )

    return distances


def travel_time_matrix(adj, candidates):
    """
    Calculate shortest travel times from every
    candidate station to every neighborhood.
    """

    matrix = {}

    for candidate in candidates:

        matrix[candidate] = dijkstra(
            adj,
            candidate
        )

    return matrix


def calculate_statistics(distances):
    """Summarize travel times from a single station."""

    reachable = [
        distance
        for distance in distances.values()
        if math.isfinite(distance)
    ]

    unreachable = len(distances) - len(reachable)

    return {
        "reachable": len(reachable),
        "unreachable": unreachable,
        "average_time": (
            sum(reachable) / len(reachable)
            if reachable else math.inf
        ),
        "maximum_time": max(
            distances.values(),
            default=math.inf
        )
    }


if __name__ == "__main__":

    # Load generated county
    with open(
        "data/sample_map_500.json",
        "r",
        encoding="utf-8"
    ) as file:

        county_map = json.load(file)

    # Create normal and rush-hour graphs
    normal_adj = build_adjacency(
        county_map,
        rush_hour=False
    )

    rush_adj = build_adjacency(
        county_map,
        rush_hour=True
    )

    # Extract candidate ambulance station IDs
    candidates = [
        node["id"]
        for node in county_map["neighborhoods"]
        if node["is_candidate"]
    ]

    # Use the first candidate station as the source
    source = candidates[0]

    print("DIJKSTRA TRAVEL TIME ANALYSIS")
    print("--------------------------------")

    print("Neighborhoods:", len(normal_adj))
    print("Candidate stations:", len(candidates))
    print("Starting station:", source)

    # Normal traffic
    start = time.perf_counter()

    normal_distances = dijkstra(
        normal_adj,
        source
    )

    normal_runtime = time.perf_counter() - start

    # Rush-hour traffic
    start = time.perf_counter()

    rush_distances = dijkstra(
        rush_adj,
        source
    )

    rush_runtime = time.perf_counter() - start

    normal_stats = calculate_statistics(
        normal_distances
    )

    rush_stats = calculate_statistics(
        rush_distances
    )

    print("\nNORMAL TRAFFIC")

    print(
        "Reachable neighborhoods:",
        normal_stats["reachable"]
    )

    print(
        "Unreachable neighborhoods:",
        normal_stats["unreachable"]
    )

    print(
        "Average travel time:",
        round(normal_stats["average_time"], 2),
        "minutes"
    )

    print(
        "Maximum travel time:",
        round(normal_stats["maximum_time"], 2),
        "minutes"
    )

    print(
        "Execution time:",
        round(normal_runtime * 1000, 4),
        "ms"
    )

    print("\nRUSH-HOUR TRAFFIC")

    print(
        "Reachable neighborhoods:",
        rush_stats["reachable"]
    )

    print(
        "Unreachable neighborhoods:",
        rush_stats["unreachable"]
    )

    print(
        "Average travel time:",
        round(rush_stats["average_time"], 2),
        "minutes"
    )

    print(
        "Maximum travel time:",
        round(rush_stats["maximum_time"], 2),
        "minutes"
    )

    print(
        "Execution time:",
        round(rush_runtime * 1000, 4),
        "ms"
    )

    # Compare travel times for the first 10 neighborhoods
    print("\nTRAVEL TIME COMPARISON")
    print("--------------------------------")

    print(
        f"{'Neighborhood':<15}"
        f"{'Normal':<15}"
        f"{'Rush Hour':<15}"
    )

    for node in list(normal_adj)[:10]:

        print(
            f"{node:<15}"
            f"{normal_distances[node]:<15.2f}"
            f"{rush_distances[node]:<15.2f}"
        )

    # Verify rush-hour travel times
    valid = all(
        rush_distances[node] >= normal_distances[node]
        for node in normal_adj
    )

    print(
        "\nRush-hour verification:",
        valid
    )

    # Generate travel-time matrices
    print("\nGENERATING TRAVEL-TIME MATRICES")

    start = time.perf_counter()

    normal_matrix = travel_time_matrix(
        normal_adj,
        candidates
    )

    rush_matrix = travel_time_matrix(
        rush_adj,
        candidates
    )

    matrix_runtime = time.perf_counter() - start

    print(
        "Normal matrix dimensions:",
        len(normal_matrix),
        "x",
        len(normal_adj)
    )

    print(
        "Rush-hour matrix dimensions:",
        len(rush_matrix),
        "x",
        len(rush_adj)
    )

    print(
        "Total travel-time calculations:",
        2 * len(candidates) * len(normal_adj)
    )

    print(
        "Matrix execution time:",
        round(matrix_runtime, 4),
        "seconds"
    )
