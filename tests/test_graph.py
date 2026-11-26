
import math
import random
import itertools

import pytest

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
# MAP GENERATOR TESTS
# ==========================================

def test_neighborhood_count():

    county = generate_map(100, seed=42)

    assert len(county["neighborhoods"]) == 100


def test_candidate_count():

    county = generate_map(100, seed=42)

    candidates = [
        node for node in county["neighborhoods"]
        if node["is_candidate"]
    ]

    assert len(candidates) == 25


def test_seed_reproducibility():

    first = generate_map(100, seed=42)
    second = generate_map(100, seed=42)

    assert first == second


def test_different_seeds():

    first = generate_map(100, seed=42)
    second = generate_map(100, seed=43)

    assert first != second


def test_vulnerability_range():

    county = generate_map(100, seed=42)

    for node in county["neighborhoods"]:

        assert 0 <= node["vulnerability"] <= 1


def test_rush_hour_weights():

    county = generate_map(100, seed=42)

    for road in county["roads"]:

        assert road["normal_time"] >= 0

        assert (
            road["rush_hour_time"]
            >= road["normal_time"]
        )


def test_no_duplicate_roads():

    county = generate_map(100, seed=42)

    edges = [
        tuple(sorted(
            (road["source"], road["target"])
        ))
        for road in county["roads"]
    ]

    assert len(edges) == len(set(edges))


def test_invalid_map_size():

    with pytest.raises(ValueError):

        generate_map(0, seed=42)


# ==========================================
# TOY GRAPH
# ==========================================

def toy_graph():

    return {
        "A": [("B", 4), ("C", 8)],

        "B": [("A", 4), ("C", 2), ("D", 7)],

        "C": [
            ("A", 8),
            ("B", 2),
            ("D", 3),
            ("E", 5)
        ],

        "D": [("B", 7), ("C", 3), ("E", 2)],

        "E": [("C", 5), ("D", 2), ("F", 3)],

        "F": [("E", 3)],

        "G": [("H", 2)],

        "H": [("G", 2)]
    }


# ==========================================
# BFS AND DFS TESTS
# ==========================================

def test_bfs_components():

    components = bfs_components(toy_graph())

    assert len(components) == 2

    assert set(components[0]) == {
        "A", "B", "C", "D", "E", "F"
    }

    assert set(components[1]) == {"G", "H"}


def test_dfs_components():

    components = dfs_components(toy_graph())

    assert len(components) == 2

    assert set(components[0]) == {
        "A", "B", "C", "D", "E", "F"
    }

    assert set(components[1]) == {"G", "H"}


def test_bfs_dfs_agreement():

    for seed in range(10):

        county = generate_map(50, seed)
        adj = build_adjacency(county)

        bfs = {
            frozenset(c)
            for c in bfs_components(adj)
        }

        dfs = {
            frozenset(c)
            for c in dfs_components(adj)
        }

        assert bfs == dfs


def test_isolated_neighborhood():

    adj = {
        0: [(1, 3)],
        1: [(0, 3)],
        2: []
    }

    assert [len(c) for c in bfs_components(adj)] == [2, 1]

    assert [len(c) for c in dfs_components(adj)] == [2, 1]


def test_empty_graph():

    assert bfs_components({}) == []

    assert dfs_components({}) == []


# ==========================================
# DIJKSTRA TESTS
# ==========================================

def test_dijkstra_known_distances():

    distances = dijkstra(toy_graph(), "A")

    expected = {
        "A": 0,
        "B": 4,
        "C": 6,
        "D": 9,
        "E": 11,
        "F": 14,
        "G": math.inf,
        "H": math.inf
    }

    assert distances == expected


def test_dijkstra_invalid_source():

    with pytest.raises(ValueError):

        dijkstra(toy_graph(), "Z")


def test_dijkstra_negative_weight():

    adj = {
        0: [(1, -2)],
        1: []
    }

    with pytest.raises(ValueError):

        dijkstra(adj, 0)


def test_dijkstra_rush_hour():

    county = generate_map(50, seed=42)

    normal = build_adjacency(
        county,
        rush_hour=False
    )

    rush = build_adjacency(
        county,
        rush_hour=True
    )

    normal_times = dijkstra(normal, 0)
    rush_times = dijkstra(rush, 0)

    for node in normal:

        assert rush_times[node] >= normal_times[node]


def test_matrix_dimensions():

    county = generate_map(40, seed=42)

    adj = build_adjacency(county)

    candidates = [
        node["id"]
        for node in county["neighborhoods"]
        if node["is_candidate"]
    ]

    matrix = travel_time_matrix(adj, candidates)

    assert len(matrix) == 10

    assert all(
        len(row) == 40
        for row in matrix.values()
    )


# ==========================================
# SLOW REFERENCE SHORTEST PATH ALGORITHM
# ==========================================

def reference_shortest_paths(adj, source):

    distances = {
        node: math.inf
        for node in adj
    }

    distances[source] = 0

    # Bellman-Ford reference implementation
    for _ in range(len(adj) - 1):

        changed = False

        for node in adj:

            for neighbor, weight in adj[node]:

                new_distance = (
                    distances[node] + weight
                )

                if new_distance < distances[neighbor]:

                    distances[neighbor] = new_distance
                    changed = True

        if not changed:
            break

    return distances


def test_dijkstra_against_reference():

    for seed in range(10):

        county = generate_map(30, seed)

        for rush_hour in [False, True]:

            adj = build_adjacency(
                county,
                rush_hour=rush_hour
            )

            actual = dijkstra(adj, 0)

            expected = reference_shortest_paths(
                adj,
                0
            )

            for node in adj:

                assert math.isclose(
                    actual[node],
                    expected[node],
                    rel_tol=1e-9,
                    abs_tol=1e-9
                )


# ==========================================
# GREEDY P-CENTER TESTS
# ==========================================

def pcenter_toy_matrix():

    # Five neighborhoods on a straight road.
    # Adjacent neighborhoods are 4 minutes apart.

    positions = {
        "A": 0,
        "B": 4,
        "C": 8,
        "D": 12,
        "E": 16
    }

    return {
        candidate: {
            node: abs(
                positions[candidate] - positions[node]
            )
            for node in positions
        }
        for candidate in positions
    }


def test_pcenter_first_station():

    matrix = pcenter_toy_matrix()

    result = greedy_pcenter(matrix, 1)

    assert result["selected_stations"] == ["C"]

    assert result["maximum_time"] == 8


def test_pcenter_two_stations():

    matrix = pcenter_toy_matrix()

    result = greedy_pcenter(matrix, 2)

    assert result["selected_stations"] == ["C", "A"]

    assert result["maximum_time"] == 8


def test_pcenter_three_stations():

    matrix = pcenter_toy_matrix()

    result = greedy_pcenter(matrix, 3)

    assert result["selected_stations"] == [
        "C", "A", "E"
    ]

    assert result["maximum_time"] == 4


def test_pcenter_unique_stations():

    matrix = pcenter_toy_matrix()

    result = greedy_pcenter(matrix, 4)

    assert len(set(result["selected_stations"])) == 4


def test_pcenter_monotonic_improvement():

    matrix = pcenter_toy_matrix()

    previous = math.inf

    for p in range(1, 6):

        result = greedy_pcenter(matrix, p)

        assert result["maximum_time"] <= previous

        previous = result["maximum_time"]


def test_pcenter_invalid_budget():

    matrix = pcenter_toy_matrix()

    with pytest.raises(ValueError):

        greedy_pcenter(matrix, 0)

    with pytest.raises(ValueError):

        greedy_pcenter(matrix, 6)


# ==========================================
# BRUTE-FORCE P-CENTER VERIFICATION
# ==========================================

def brute_force_pcenter(matrix, p):

    candidates = list(matrix)
    neighborhoods = list(matrix[candidates[0]])

    best_stations = None
    best_time = math.inf

    # Examine every possible station combination
    for stations in itertools.combinations(
        candidates, p
    ):

        maximum_time = max(
            min(
                matrix[station][node]
                for station in stations
            )
            for node in neighborhoods
        )

        if (
            best_stations is None
            or maximum_time < best_time
        ):

            best_time = maximum_time
            best_stations = stations

    return best_stations, best_time


def test_pcenter_against_brute_force():

    matrix = pcenter_toy_matrix()

    for p in range(1, 6):

        greedy = greedy_pcenter(matrix, p)

        optimal_stations, optimal_time = (
            brute_force_pcenter(matrix, p)
        )

        assert (
            greedy["maximum_time"] >= optimal_time
        )


def test_pcenter_known_greedy_limitation():

    matrix = pcenter_toy_matrix()

    greedy = greedy_pcenter(matrix, 2)

    optimal_stations, optimal_time = (
        brute_force_pcenter(matrix, 2)
    )

    assert greedy["maximum_time"] == 8

    assert optimal_time == 4


def test_random_pcenter_against_brute_force():

    for seed in range(10):

        rng = random.Random(seed)

        positions = {
            i: rng.uniform(0, 50)
            for i in range(8)
        }

        matrix = {
            candidate: {
                node: abs(
                    positions[candidate] - positions[node]
                )
                for node in positions
            }
            for candidate in positions
        }

        for p in [1, 2, 3]:

            greedy = greedy_pcenter(matrix, p)

            optimal_stations, optimal_time = (
                brute_force_pcenter(matrix, p)
            )

            assert (
                greedy["maximum_time"]
                >= optimal_time - 1e-9
            )

            assert len(
                greedy["selected_stations"]
            ) == p

            for node in positions:

                expected = min(
                    matrix[station][node]
                    for station in greedy["selected_stations"]
                )

                assert math.isclose(
                    greedy["nearest_times"][node],
                    expected
                )
