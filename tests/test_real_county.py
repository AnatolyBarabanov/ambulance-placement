
import unittest
import sys
import math
import io

import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "data"))

from real_county import (
    load_county_data,
    classify_neighborhoods,
    select_candidates
)

from real_travel_times import (
    build_adjacency_list,
    dijkstra,
    calculate_access_time,
    calculate_travel_matrix
)

from real_optimization import (
    load_data,
    greedy_p_center,
    evaluate_placement,
    compare_placements
)


class RealCountyFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Suppress repeated loading output
        with redirect_stdout(io.StringIO()):

            cls.neighborhoods, cls.candidates, \
                cls.stations, cls.matrices = load_data()


class TestRealCountyData(RealCountyFixture):

    def test_neighborhood_count(self):

        self.assertEqual(
            len(self.neighborhoods), 594
        )

    def test_existing_station_count(self):

        self.assertEqual(
            len(self.stations), 40
        )

    def test_candidate_count(self):

        self.assertEqual(
            len(self.candidates), 148
        )

    def test_total_population(self):

        self.assertEqual(
            int(self.neighborhoods["population"].sum()),
            867573
        )

    def test_unique_neighborhood_ids(self):

        self.assertTrue(
            self.neighborhoods["id"].is_unique
        )

    def test_unique_census_geoids(self):

        self.assertTrue(
            self.neighborhoods["GEOID"].is_unique
        )

    def test_unique_station_ids(self):

        self.assertTrue(
            self.stations["station_id"].is_unique
        )

    def test_positive_population(self):

        self.assertTrue(
            (self.neighborhoods["population"] > 0).all()
        )

    def test_valid_vulnerability(self):

        values = self.neighborhoods["vulnerability"]

        self.assertTrue(values.notna().all())

        self.assertTrue(
            values.between(0, 1).all()
        )

    def test_valid_coordinates(self):

        for dataset in [
            self.neighborhoods,
            self.stations,
            self.candidates
        ]:

            self.assertTrue(
                dataset["latitude"].between(-90, 90).all()
            )

            self.assertTrue(
                dataset["longitude"].between(-180, 180).all()
            )

    def test_area_classification(self):

        expected = {
            "urban": 228,
            "suburban": 304,
            "rural": 62
        }

        actual = (
            self.neighborhoods["area_type"]
            .value_counts()
            .to_dict()
        )

        self.assertEqual(actual, expected)

    def test_candidate_distribution(self):

        expected = {
            "urban": 57,
            "suburban": 76,
            "rural": 15
        }

        actual = (
            self.candidates["area_type"]
            .value_counts()
            .to_dict()
        )

        self.assertEqual(actual, expected)

    def test_candidates_are_neighborhoods(self):

        self.assertTrue(
            set(self.candidates["id"]).issubset(
                set(self.neighborhoods["id"])
            )
        )

    def test_reproducible_candidate_selection(self):

        with redirect_stdout(io.StringIO()):

            _, new_candidates = select_candidates(
                self.neighborhoods,
                seed=42
            )

        self.assertEqual(
            set(new_candidates["id"]),
            set(self.candidates["id"])
        )


class TestRealRoadNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        with redirect_stdout(io.StringIO()):

            cls.neighborhoods, cls.stations, cls.graph = (
                load_county_data()
            )

            cls.adjacency = build_adjacency_list(
                cls.graph
            )

    def test_road_node_count(self):

        self.assertEqual(
            self.graph.number_of_nodes(), 26376
        )

    def test_road_edge_count(self):

        self.assertEqual(
            self.graph.number_of_edges(), 69682
        )

    def test_directed_graph(self):

        self.assertTrue(
            self.graph.is_directed()
        )

    def test_strong_connectivity(self):

        self.assertTrue(
            nx.is_strongly_connected(self.graph)
        )

    def test_single_strong_component(self):

        self.assertEqual(
            nx.number_strongly_connected_components(
                self.graph
            ),
            1
        )

    def test_neighborhood_road_connections(self):

        self.assertTrue(
            self.neighborhoods["road_node"]
            .isin(self.graph.nodes)
            .all()
        )

    def test_station_road_connections(self):

        self.assertTrue(
            self.stations["road_node"]
            .isin(self.graph.nodes)
            .all()
        )

    def test_population_preserved(self):

        self.assertEqual(
            int(self.neighborhoods["population"].sum()),
            867573
        )

    def test_valid_snapping_distances(self):

        for dataset in [
            self.neighborhoods,
            self.stations
        ]:

            distances = dataset["snap_distance_m"]

            self.assertTrue(
                np.isfinite(distances).all()
            )

            self.assertTrue(
                (distances >= 0).all()
            )

    def test_adjacency_node_count(self):

        self.assertEqual(
            len(self.adjacency),
            self.graph.number_of_nodes()
        )

    def test_adjacency_edge_count(self):

        total_edges = sum(
            len(edges)
            for edges in self.adjacency.values()
        )

        self.assertEqual(
            total_edges,
            self.graph.number_of_edges()
        )

    def test_valid_road_weights(self):

        for edges in self.adjacency.values():

            for target, normal, rush in edges:

                self.assertTrue(math.isfinite(normal))

                self.assertTrue(math.isfinite(rush))

                self.assertGreaterEqual(normal, 0)

                self.assertGreaterEqual(rush, normal)

    def test_real_dijkstra_against_networkx(self):

        source = int(
            self.stations.iloc[0]["road_node"]
        )

        actual = dijkstra(
            self.adjacency,
            source,
            rush_hour=False
        )

        expected = nx.single_source_dijkstra_path_length(
            self.graph,
            source,
            weight="travel_time"
        )

        self.assertEqual(
            len(actual), len(expected)
        )

        for node, seconds in expected.items():

            self.assertAlmostEqual(
                actual[node],
                seconds / 60,
                places=7
            )

class TestRealDijkstra(unittest.TestCase):

    def setUp(self):

        self.adjacency = {

            0: [
                (1, 4, 6),
                (2, 1, 2)
            ],

            1: [
                (3, 1, 2)
            ],

            2: [
                (1, 2, 3),
                (3, 5, 8)
            ],

            3: []
        }

    def test_source_distance_zero(self):

        result = dijkstra(self.adjacency, 0)

        self.assertEqual(result[0], 0)

    def test_shortest_path_normal(self):

        result = dijkstra(self.adjacency, 0)

        expected = {
            0: 0,
            1: 3,
            2: 1,
            3: 4
        }

        self.assertEqual(result, expected)

    def test_shortest_path_rush(self):

        result = dijkstra(
            self.adjacency,
            0,
            rush_hour=True
        )

        expected = {
            0: 0,
            1: 5,
            2: 2,
            3: 7
        }

        self.assertEqual(result, expected)

    def test_directed_edges(self):

        result = dijkstra(self.adjacency, 3)

        self.assertTrue(math.isinf(result[0]))

    def test_unreachable_node(self):

        adjacency = {
            0: [(1, 2, 3)],
            1: [],
            2: []
        }

        result = dijkstra(adjacency, 0)

        self.assertTrue(math.isinf(result[2]))

    def test_parallel_edges(self):

        adjacency = {
            0: [
                (1, 10, 15),
                (1, 3, 5)
            ],
            1: []
        }

        result = dijkstra(adjacency, 0)

        self.assertEqual(result[1], 3)

    def test_single_node(self):

        result = dijkstra({0: []}, 0)

        self.assertEqual(result, {0: 0})

    def test_zero_weight_edge(self):

        adjacency = {
            0: [(1, 0, 0)],
            1: []
        }

        result = dijkstra(adjacency, 0)

        self.assertEqual(result[1], 0)

    def test_access_time_conversion(self):

        self.assertAlmostEqual(
            calculate_access_time(1000),
            2.0
        )

    def test_zero_access_distance(self):

        self.assertEqual(
            calculate_access_time(0),
            0
        )

class TestRealTravelMatrices(RealCountyFixture):

    def test_existing_normal_shape(self):

        self.assertEqual(
            self.matrices["existing_normal"].shape,
            (40, 594)
        )

    def test_existing_rush_shape(self):

        self.assertEqual(
            self.matrices["existing_rush"].shape,
            (40, 594)
        )

    def test_candidate_normal_shape(self):

        self.assertEqual(
            self.matrices["candidate_normal"].shape,
            (148, 594)
        )

    def test_candidate_rush_shape(self):

        self.assertEqual(
            self.matrices["candidate_rush"].shape,
            (148, 594)
        )

    def test_total_travel_values(self):

        total = sum(
            matrix.size
            for matrix in self.matrices.values()
        )

        self.assertEqual(total, 223344)

    def test_no_unreachable_destinations(self):

        for matrix in self.matrices.values():

            self.assertTrue(
                np.isfinite(matrix).all()
            )

    def test_nonnegative_travel_times(self):

        for matrix in self.matrices.values():

            self.assertTrue(
                (matrix >= 0).all()
            )

    def test_existing_traffic_consistency(self):

        self.assertTrue(
            np.all(
                self.matrices["existing_rush"]
                >= self.matrices["existing_normal"] - 1e-9
            )
        )

    def test_candidate_traffic_consistency(self):

        self.assertTrue(
            np.all(
                self.matrices["candidate_rush"]
                >= self.matrices["candidate_normal"] - 1e-9
            )
        )

    def test_candidate_self_travel_time(self):

        neighborhood_ids = (
            self.neighborhoods["id"].tolist()
        )

        positions = {
            neighborhood_id: index
            for index, neighborhood_id
            in enumerate(neighborhood_ids)
        }

        for row, candidate_id in enumerate(
            self.candidates["id"]
        ):

            column = positions[candidate_id]

            self.assertEqual(
                self.matrices["candidate_normal"][
                    row, column
                ],
                0
            )

            self.assertEqual(
                self.matrices["candidate_rush"][
                    row, column
                ],
                0
            )

class TestGreedyPCenter(unittest.TestCase):

    def setUp(self):

        self.matrix = np.array([

            [0, 2, 8, 10],

            [2, 0, 6, 8],

            [8, 6, 0, 2],

            [10, 8, 2, 0]

        ], dtype=float)

    def optimize(self, budget):

        with redirect_stdout(io.StringIO()):

            return greedy_p_center(
                self.matrix,
                budget
            )

    def test_single_station(self):

        selected, times, history = self.optimize(1)

        self.assertEqual(len(selected), 1)

        self.assertEqual(times.max(), 8)

    def test_two_station_solution(self):

        selected, times, history = self.optimize(2)

        self.assertEqual(len(selected), 2)

        self.assertEqual(times.max(), 2)

    def test_all_candidates(self):

        selected, times, history = self.optimize(4)

        self.assertEqual(times.max(), 0)

    def test_no_duplicate_selection(self):

        selected, times, history = self.optimize(4)

        self.assertEqual(
            len(selected),
            len(set(selected))
        )

    def test_invalid_zero_budget(self):

        with self.assertRaises(ValueError):

            self.optimize(0)

    def test_invalid_large_budget(self):

        with self.assertRaises(ValueError):

            self.optimize(5)

    def test_monotonic_maximum_time(self):

        selected, times, history = self.optimize(4)

        self.assertTrue(
            np.all(
                np.diff(history["maximum_time"]) <= 1e-9
            )
        )

    def test_monotonic_average_time(self):

        selected, times, history = self.optimize(4)

        self.assertTrue(
            np.all(
                np.diff(history["average_time"]) <= 1e-9
            )
        )

    def test_nearest_time_correctness(self):

        selected, times, history = self.optimize(2)

        expected = self.matrix[selected].min(axis=0)

        np.testing.assert_allclose(times, expected)

    def test_deterministic_selection(self):

        first, _, _ = self.optimize(2)

        second, _, _ = self.optimize(2)

        self.assertEqual(first, second)

class TestCoverageMetrics(unittest.TestCase):

    def setUp(self):

        self.neighborhoods = pd.DataFrame({

            "population": [100, 200, 700],

            "vulnerability": [0.1, 0.2, 0.3]

        })

        self.times = np.array([4.0, 8.0, 10.0])

        self.metrics = evaluate_placement(
            self.times,
            self.neighborhoods,
            threshold=8
        )

    def test_average_time(self):

        self.assertAlmostEqual(
            self.metrics["average_time"],
            22 / 3
        )

    def test_maximum_time(self):

        self.assertEqual(
            self.metrics["maximum_time"],
            10
        )

    def test_population_coverage(self):

        self.assertAlmostEqual(
            self.metrics["population_coverage"],
            30
        )

    def test_neighborhood_coverage(self):

        self.assertAlmostEqual(
            self.metrics["neighborhood_coverage"],
            2 / 3 * 100
        )

    def test_covered_population(self):

        self.assertEqual(
            self.metrics["covered_population"],
            300
        )

    def test_uncovered_population(self):

        self.assertEqual(
            self.metrics["uncovered_population"],
            700
        )

    def test_population_weighted_time(self):

        self.assertAlmostEqual(
            self.metrics["population_weighted_time"],
            9.0
        )

    def test_vulnerability_weighted_coverage(self):

        self.assertAlmostEqual(
            self.metrics["vulnerability_weighted_coverage"],
            50 / 260 * 100
        )

    def test_threshold_inclusive(self):

        self.assertEqual(
            self.metrics["covered_population"],
            300
        )

class TestRealOptimization(RealCountyFixture):

    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        with redirect_stdout(io.StringIO()):

            cls.normal_selected, cls.normal_times, \
                cls.normal_history = greedy_p_center(
                    cls.matrices["candidate_normal"],
                    40
                )

            cls.rush_selected, cls.rush_times, \
                cls.rush_history = greedy_p_center(
                    cls.matrices["candidate_rush"],
                    40
                )

        cls.results, cls.placements = compare_placements(
            cls.neighborhoods,
            cls.matrices,
            cls.normal_selected,
            cls.rush_selected
        )

    def test_normal_station_budget(self):

        self.assertEqual(
            len(set(self.normal_selected)),
            40
        )

    def test_rush_station_budget(self):

        self.assertEqual(
            len(set(self.rush_selected)),
            40
        )

    def test_normal_optimality(self):

        lower_bound = (
            self.matrices["candidate_normal"]
            .min(axis=0)
            .max()
        )

        self.assertAlmostEqual(
            self.normal_times.max(),
            lower_bound,
            places=7
        )

    def test_rush_optimality(self):

        lower_bound = (
            self.matrices["candidate_rush"]
            .min(axis=0)
            .max()
        )

        self.assertAlmostEqual(
            self.rush_times.max(),
            lower_bound,
            places=7
        )

    def test_normal_verified_result(self):

        self.assertAlmostEqual(
            self.normal_times.max(),
            9.4071,
            places=4
        )

    def test_rush_verified_result(self):

        self.assertAlmostEqual(
            self.rush_times.max(),
            12.4307,
            places=4
        )

    def test_normal_improves_existing_maximum(self):

        baseline = (
            self.matrices["existing_normal"]
            .min(axis=0)
            .max()
        )

        self.assertLess(
            self.normal_times.max(),
            baseline
        )

    def test_rush_improves_existing_maximum(self):

        baseline = (
            self.matrices["existing_rush"]
            .min(axis=0)
            .max()
        )

        self.assertLess(
            self.rush_times.max(),
            baseline
        )

    def test_normal_optimal_budget(self):

        lower_bound = (
            self.matrices["candidate_normal"]
            .min(axis=0)
            .max()
        )

        matching = self.normal_history[
            np.isclose(
                self.normal_history["maximum_time"],
                lower_bound,
                atol=1e-8,
                rtol=0
            )
        ]

        self.assertEqual(
            int(matching.iloc[0]["iteration"]),
            25
        )

    def test_rush_optimal_budget(self):

        lower_bound = (
            self.matrices["candidate_rush"]
            .min(axis=0)
            .max()
        )

        matching = self.rush_history[
            np.isclose(
                self.rush_history["maximum_time"],
                lower_bound,
                atol=1e-8,
                rtol=0
            )
        ]

        self.assertEqual(
            int(matching.iloc[0]["iteration"]),
            29
        )

    def test_six_comparison_scenarios(self):

        self.assertEqual(
            len(self.results),
            6
        )

    def test_coverage_percentage_bounds(self):

        self.assertTrue(
            self.results["population_coverage"]
            .between(0, 100)
            .all()
        )

    def test_rural_rush_coverage_improvement(self):

        rural = (
            self.neighborhoods["area_type"]
            .to_numpy() == "rural"
        )

        rural_data = self.neighborhoods.loc[rural]

        baseline = evaluate_placement(
            self.placements["Existing EMS"]["rush"][rural],
            rural_data
        )

        optimized = evaluate_placement(
            self.placements["Rush Optimized"]["rush"][rural],
            rural_data
        )

        self.assertGreater(
            optimized["population_coverage"],
            baseline["population_coverage"]
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
