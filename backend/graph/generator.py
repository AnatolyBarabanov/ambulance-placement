
"""
Problem:
Generate a synthetic county road network for ambulance station
placement and emergency travel-time analysis.

Input:
    n: Number of neighborhoods.
    seed: Random seed for reproducibility.

Output:
    A dictionary containing neighborhoods and weighted roads.

Design:
    Generate urban, suburban and rural neighborhoods.
    Connect each neighborhood to its three nearest neighbors.
    Calculate normal and rush-hour travel times.

Time Complexity:
    O(n^2 log n), due to sorting neighboring distances
    for every neighborhood.

Space Complexity:
    O(n), because each neighborhood selects at most three roads
    and only one distance list is stored at a time.
"""

import random
import math
import json
from pathlib import Path


# County configuration
COUNTY_SIZE = 50
CENTER = COUNTY_SIZE / 2

AREA_SETTINGS = {
    "urban": {
        "population": (1500, 5000),
        "speed": 35,
        "traffic": 1.8,
        "uninsured": (0.05, 0.15),
        "elderly": (0.08, 0.18),
        "low_income": (0.10, 0.30)
    },
    "suburban": {
        "population": (500, 2500),
        "speed": 50,
        "traffic": 1.4,
        "uninsured": (0.03, 0.12),
        "elderly": (0.10, 0.22),
        "low_income": (0.05, 0.20)
    },
    "rural": {
        "population": (100, 800),
        "speed": 70,
        "traffic": 1.1,
        "uninsured": (0.10, 0.25),
        "elderly": (0.18, 0.35),
        "low_income": (0.15, 0.40)
    }
}


def generate_position(area_type, rng):
    """Generate coordinates based on neighborhood type."""

    if area_type == "urban":
        minimum, maximum = 0, 8

    elif area_type == "suburban":
        minimum, maximum = 8, 17

    else:
        minimum, maximum = 17, 25

    angle = rng.uniform(0, 2 * math.pi)

    radius = math.sqrt(
        rng.uniform(minimum ** 2, maximum ** 2)
    )

    x = CENTER + radius * math.cos(angle)
    y = CENTER + radius * math.sin(angle)

    return round(x, 4), round(y, 4)


def calculate_distance(node1, node2):
    """Calculate Euclidean distance in kilometers."""

    return math.hypot(
        node2["x"] - node1["x"],
        node2["y"] - node1["y"]
    )


def calculate_vulnerability(area_type, rng):
    """Calculate a synthetic vulnerability score."""

    settings = AREA_SETTINGS[area_type]

    uninsured = rng.uniform(*settings["uninsured"])
    elderly = rng.uniform(*settings["elderly"])
    low_income = rng.uniform(*settings["low_income"])

    vulnerability = (
        uninsured + elderly + low_income
    ) / 3

    return {
        "uninsured": round(uninsured, 4),
        "elderly": round(elderly, 4),
        "low_income": round(low_income, 4),
        "vulnerability": round(vulnerability, 4)
    }


def generate_map(n, seed=42):
    """Generate a reproducible synthetic county."""

    if n < 1:
        raise ValueError("Number of neighborhoods must be positive.")

    rng = random.Random(seed)

    neighborhoods = []
    roads = []

    # Determine neighborhood distribution
    urban_count = round(n * 0.50)
    suburban_count = round(n * 0.35)
    rural_count = n - urban_count - suburban_count

    area_types = (
        ["urban"] * urban_count
        + ["suburban"] * suburban_count
        + ["rural"] * rural_count
    )

    rng.shuffle(area_types)

    # Select approximately 25% as candidate stations
    candidate_count = max(1, round(n * 0.25))

    candidate_ids = set(
        rng.sample(range(n), candidate_count)
    )

    # Generate neighborhoods
    for i in range(n):

        area_type = area_types[i]
        settings = AREA_SETTINGS[area_type]

        x, y = generate_position(area_type, rng)

        population = rng.randint(
            *settings["population"]
        )

        vulnerability = calculate_vulnerability(
            area_type, rng
        )

        
        neighborhood = {
            "id": i,
            "x": x,
            "y": y,
            "area": area_type,
            "area_type": area_type,
            "population": population,
            "is_candidate": i in candidate_ids,
            **vulnerability
        }


        neighborhoods.append(neighborhood)

    # Prevent duplicate undirected roads
    existing_roads = set()

    # Connect each neighborhood to its three nearest neighbors
    for node in neighborhoods:

        distances = []

        for other in neighborhoods:

            if node["id"] == other["id"]:
                continue

            distance = calculate_distance(node, other)

            distances.append(
                (distance, other["id"])
            )

        distances.sort()

        for distance, neighbor_id in distances[:3]:

            road_id = tuple(sorted(
                (node["id"], neighbor_id)
            ))

            if road_id in existing_roads:
                continue

            existing_roads.add(road_id)

            neighbor = neighborhoods[neighbor_id]

            settings1 = AREA_SETTINGS[node["area_type"]]
            settings2 = AREA_SETTINGS[neighbor["area_type"]]

            # Average road speed between both neighborhoods
            speed = (
                settings1["speed"] + settings2["speed"]
            ) / 2

            # Average congestion factor
            traffic_factor = (
                settings1["traffic"] + settings2["traffic"]
            ) / 2

            # Convert hours into minutes
            normal_time = (distance / speed) * 60

            rush_hour_time = normal_time * traffic_factor

            
            road = {
                "u": road_id[0],
                "v": road_id[1],
                "minutes": round(normal_time, 4),
                "rush_minutes": round(rush_hour_time, 4),
                "source": road_id[0],
                "target": road_id[1],
                "distance_km": round(distance, 4),
                "speed_kmh": speed,
                "normal_time": round(normal_time, 4),
                "rush_hour_time": round(rush_hour_time, 4)
            }


            roads.append(road)

    return {
        "name": "Synthetic County",
        "seed": seed,
        "county_size_km": COUNTY_SIZE,
        "neighborhoods": neighborhoods,
        "roads": roads
    }


def save_map(county_map, filename):
    """Save the generated county as a JSON file."""

    path = Path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", encoding="utf-8") as file:

        json.dump(county_map, file, indent=4)

    print(f"Map saved to: {path}")


if __name__ == "__main__":

    county_map = generate_map(500, seed=42)

    save_map(
        county_map,
        "data/sample_map_500.json"
    )

    neighborhoods = county_map["neighborhoods"]
    roads = county_map["roads"]

    print("\nCOUNTY MAP SUMMARY")
    print("----------------------------")

    print("Neighborhoods:", len(neighborhoods))
    print("Roads:", len(roads))

    print(
        "Total population:",
        sum(node["population"] for node in neighborhoods)
    )

    print(
        "Candidate stations:",
        sum(node["is_candidate"] for node in neighborhoods)
    )

    for area in AREA_SETTINGS:

        count = sum(
            node["area_type"] == area
            for node in neighborhoods
        )

        print(f"{area.capitalize()} neighborhoods: {count}")
