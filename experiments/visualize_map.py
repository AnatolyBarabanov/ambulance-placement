
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


with open("data/sample_map_500.json", "r") as file:
    county_map = json.load(file)


neighborhoods = county_map["neighborhoods"]
roads = county_map["roads"]

nodes = {
    node["id"]: node
    for node in neighborhoods
}

colors = {
    "urban": "royalblue",
    "suburban": "orange",
    "rural": "forestgreen"
}

plt.figure(figsize=(10, 10))

# Draw road connections
for road in roads:

    source = nodes[road["source"]]
    target = nodes[road["target"]]

    plt.plot(
        [source["x"], target["x"]],
        [source["y"], target["y"]],
        color="gray",
        linewidth=0.5,
        alpha=0.4
    )

# Draw neighborhoods
for node in neighborhoods:

    marker = "*" if node["is_candidate"] else "o"

    size = 65 if node["is_candidate"] else 15

    plt.scatter(
        node["x"],
        node["y"],
        color=colors[node["area_type"]],
        marker=marker,
        s=size
    )

legend_items = [
    Line2D([0], [0], marker="o", color="royalblue",
           linestyle="None", label="Urban"),

    Line2D([0], [0], marker="o", color="orange",
           linestyle="None", label="Suburban"),

    Line2D([0], [0], marker="o", color="forestgreen",
           linestyle="None", label="Rural"),

    Line2D([0], [0], marker="*", color="black",
           linestyle="None", label="Candidate station"),

    Line2D([0], [0], color="gray",
           label="Road")
]

plt.legend(handles=legend_items)

plt.title("Synthetic County - 500 Neighborhoods")
plt.xlabel("X Coordinate (km)")
plt.ylabel("Y Coordinate (km)")

plt.xlim(0, 50)
plt.ylim(0, 50)

plt.gca().set_aspect("equal")

plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    "data/sample_map_500.png",
    dpi=300
)

plt.show()
