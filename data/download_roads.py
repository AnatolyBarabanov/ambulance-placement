
"""
Montgomery County Road Network Collection

Purpose:
    Download the real drivable road network from
    OpenStreetMap and calculate estimated travel times.

Output:
    montgomery_roads.graphml

Road network:
    Nodes represent road intersections and endpoints.
    Edges represent road segments.
    Edge weights represent estimated travel times.
"""

import osmnx as ox
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY = "Montgomery County, Pennsylvania, USA"

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.requests_timeout = 600


def download_road_network():

    print("MONTGOMERY COUNTY ROAD NETWORK")
    print("=" * 50)

    print("\nDownloading OpenStreetMap roads...")

    graph = ox.graph_from_place(
        COUNTY,
        network_type="drive",
        simplify=True,
        retain_all=True
    )

    print("\nCalculating road speeds...")

    graph = ox.routing.add_edge_speeds(graph)

    print("Calculating road travel times...")

    graph = ox.routing.add_edge_travel_times(graph)

    output_file = OUTPUT_DIR / "montgomery_roads.graphml"

    ox.save_graphml(
        graph,
        filepath=output_file
    )

    print("\nROAD NETWORK SUMMARY")
    print("-" * 40)

    print("County:", COUNTY)
    print("Road nodes:", graph.number_of_nodes())
    print("Road edges:", graph.number_of_edges())
    print("Directed graph:", graph.is_directed())

    total_length = sum(
        data.get("length", 0)
        for _, _, data in graph.edges(data=True)
    )

    print("Total directed edge length (km):",
          round(total_length / 1000, 2))

    print("\nFIRST FIVE ROAD SEGMENTS")
    print("-" * 40)

    for source, target, data in list(
        graph.edges(data=True)
    )[:5]:

        print(
            f"Road: {source} -> {target} | "
            f"Distance: {data.get('length', 0):.2f} m | "
            f"Speed: {data.get('speed_kph', 0):.2f} km/h | "
            f"Travel time: "
            f"{data.get('travel_time', 0):.2f} seconds"
        )

    print("\nSaved to:", output_file)

    return graph


if __name__ == "__main__":

    download_road_network()
