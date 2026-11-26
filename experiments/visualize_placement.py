
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from backend.graph.traversal import build_adjacency
from backend.graph.dijkstra import travel_time_matrix
from backend.graph.pcenter import greedy_pcenter


# Experiment settings
STATION_BUDGET = 20
TARGET_TIME = 8
RUSH_HOUR = False


# Load synthetic county
with open("data/sample_map_500.json", "r") as file:
    county_map = json.load(file)


neighborhoods = county_map["neighborhoods"]
roads = county_map["roads"]

nodes = {
    node["id"]: node
    for node in neighborhoods
}


# Extract candidate stations
candidates = [
    node["id"]
    for node in neighborhoods
    if node["is_candidate"]
]


# Build graph and calculate travel times
adj = build_adjacency(
    county_map,
    rush_hour=RUSH_HOUR
)

matrix = travel_time_matrix(adj, candidates)


# Run greedy p-center
result = greedy_pcenter(
    matrix,
    STATION_BUDGET
)

selected = set(result["selected_stations"])
nearest_times = result["nearest_times"]


plt.figure(figsize=(11, 10))


# Draw roads
for road in roads:

    source = nodes[road["source"]]
    target = nodes[road["target"]]

    plt.plot(
        [source["x"], target["x"]],
        [source["y"], target["y"]],
        color="gray",
        linewidth=0.5,
        alpha=0.3
    )


# Draw neighborhoods according to travel time
for node in neighborhoods:

    travel_time = nearest_times[node["id"]]

    if travel_time <= TARGET_TIME:
        color = "forestgreen"

    elif travel_time <= TARGET_TIME * 2:
        color = "orange"

    else:
        color = "red"

    plt.scatter(
        node["x"],
        node["y"],
        color=color,
        s=20,
        zorder=2
    )


# Draw selected ambulance stations
for station in selected:

    node = nodes[station]

    plt.scatter(
        node["x"],
        node["y"],
        marker="*",
        color="royalblue",
        edgecolors="black",
        s=220,
        zorder=3
    )


legend_items = [
    Line2D(
        [0], [0],
        marker="o",
        color="forestgreen",
        linestyle="None",
        label="Within 8 minutes"
    ),

    Line2D(
        [0], [0],
        marker="o",
        color="orange",
        linestyle="None",
        label="Over 8 to 16 minutes"
    ),

    Line2D(
        [0], [0],
        marker="o",
        color="red",
        linestyle="None",
        label="Over 16 minutes"
    ),

    Line2D(
        [0], [0],
        marker="*",
        color="royalblue",
        markersize=14,
        linestyle="None",
        label="Selected ambulance station"
    ),

    Line2D(
        [0], [0],
        color="gray",
        label="Road"
    )
]


plt.legend(handles=legend_items)

traffic = "Rush Hour" if RUSH_HOUR else "Normal Traffic"

plt.title(
    f"Greedy P-Center: {STATION_BUDGET} Ambulance Stations\n"
    f"{traffic} | "
    f"Maximum Travel Time: {result['maximum_time']:.2f} min"
)

plt.xlabel("X Coordinate (km)")
plt.ylabel("Y Coordinate (km)")

plt.xlim(0, 50)
plt.ylim(0, 50)

plt.gca().set_aspect("equal")
plt.grid(alpha=0.2)
plt.tight_layout()


traffic_name = "rush" if RUSH_HOUR else "normal"

filename = (
    f"data/pcenter_{STATION_BUDGET}_{traffic_name}.png"
)

plt.savefig(filename, dpi=300)

print("Visualization saved:", filename)
print("Selected stations:", result["selected_stations"])
print("Maximum travel time:", result["maximum_time"])

plt.show()
