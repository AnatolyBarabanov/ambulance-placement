
import numpy as np
import pandas as pd

from real_optimization import load_data, evaluate_placement


def analyze_candidate_limit(matrix, neighborhoods, traffic):

    print(f"\n{traffic.upper()} CANDIDATE LIMIT")
    print("=" * 50)

    # Best possible travel time for each neighborhood
    # if every candidate station were available
    best_possible = matrix.min(axis=0)

    lower_bound = best_possible.max()

    worst_index = np.argmax(best_possible)

    worst_neighborhood = neighborhoods.iloc[worst_index]

    nearest_candidate = np.argmin(
        matrix[:, worst_index]
    )

    print("Available candidates:", matrix.shape[0])

    print(
        "Lowest possible maximum travel time:",
        round(lower_bound, 4),
        "minutes"
    )

    print(
        "Worst-served neighborhood ID:",
        worst_neighborhood["id"]
    )

    print("GEOID:", worst_neighborhood["GEOID"])

    print("Area type:", worst_neighborhood["area_type"])

    print("Population:", worst_neighborhood["population"])

    print(
        "Nearest candidate row:",
        nearest_candidate
    )

    metrics = evaluate_placement(
        best_possible,
        neighborhoods
    )

    print(
        "Maximum possible population coverage:",
        round(metrics["population_coverage"], 2),
        "%"
    )

    return lower_bound


def main():

    neighborhoods, candidates, stations, matrices = (
        load_data()
    )

    for traffic in ["normal", "rush"]:

        matrix = matrices[f"candidate_{traffic}"]

        lower_bound = analyze_candidate_limit(
            matrix,
            neighborhoods,
            traffic
        )

        history = pd.read_csv(
            f"data/processed/optimization/"
            f"{traffic}_optimization_history.csv"
        )

        greedy_result = history.iloc[-1]["maximum_time"]

        gap = greedy_result - lower_bound

        print("\nOPTIMALITY ANALYSIS")
        print("-" * 40)

        print(
            "Greedy maximum:",
            round(greedy_result, 4)
        )

        print(
            "Theoretical lower bound:",
            round(lower_bound, 4)
        )

        print(
            "Absolute optimality gap:",
            round(gap, 6)
        )

        print(
            "Percentage optimality gap:",
            round(gap / lower_bound * 100, 4),
            "%"
        )

        if np.isclose(
            greedy_result,
            lower_bound,
            atol=1e-8,
            rtol=0
        ):

            print(
                "RESULT: Greedy solution achieved "
                "the theoretical minimum maximum travel time."
            )

        else:

            print(
                "RESULT: Greedy solution is above "
                "the theoretical lower bound."
            )

        matching = history[
            np.isclose(
                history["maximum_time"],
                lower_bound,
                atol=1e-8,
                rtol=0
            )
        ]

        if not matching.empty:

            print(
                "First greedy station budget reaching lower bound:",
                int(matching.iloc[0]["iteration"])
            )


if __name__ == "__main__":

    main()
