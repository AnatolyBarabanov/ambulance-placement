
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_county_data():

    print("REAL MONTGOMERY COUNTY")
    print("=" * 50)

    # Load census geographic boundaries/locations
    neighborhoods = gpd.read_file(
        DATA_DIR / "census_block_groups.geojson"
    )

    # Load validated demographic data
    demographics = pd.read_csv(
        DATA_DIR / "census_demographics_validated.csv",
        dtype={"GEOID": str}
    )

    # Load existing EMS station locations
    stations = gpd.read_file(
        DATA_DIR / "ems_stations_cleaned.geojson"
    )

    # Load the OpenStreetMap road network
    graph = ox.load_graphml(
        DATA_DIR / "montgomery_roads.graphml"
    )

    print("\nDATA LOADED")
    print("-" * 40)

    print("Census block groups:", len(neighborhoods))
    print("Demographic records:", len(demographics))
    print("EMS stations:", len(stations))
    print("Road nodes:", graph.number_of_nodes())
    print("Road edges:", graph.number_of_edges())

    # Ensure matching GEOID data types
    neighborhoods["GEOID"] = (
        neighborhoods["GEOID"].astype(str)
    )

    # Merge geographic and demographic data
    neighborhoods = neighborhoods.merge(
        demographics,
        on="GEOID",
        how="left",
        validate="one_to_one"
    )

    print("\nMERGED DATA")
    print("-" * 40)

    print("Total neighborhoods:", len(neighborhoods))
    print("Missing population:",
          neighborhoods["population"].isna().sum())

    if neighborhoods["population"].isna().any():
        raise ValueError("Some GEOIDs failed to merge.")

    # Keep populated neighborhoods as demand locations
    neighborhoods = neighborhoods[
        neighborhoods["population"] > 0
    ].copy()

    neighborhoods.reset_index(drop=True, inplace=True)

    print("Populated neighborhoods:", len(neighborhoods))
    print("Total population:",
          int(neighborhoods["population"].sum()))

    print("Missing vulnerability:",
          neighborhoods["vulnerability"].isna().sum())

    # Convert locations to geographic coordinates
    neighborhoods = neighborhoods.to_crs(epsg=4326)
    stations = stations.to_crs(epsg=4326)

    # Handle either polygon or point geometry
    if not neighborhoods.geometry.geom_type.eq("Point").all():

        neighborhoods = neighborhoods.to_crs(epsg=26918)

        neighborhoods["geometry"] = (
            neighborhoods.geometry.centroid
        )

        neighborhoods = neighborhoods.to_crs(epsg=4326)

    # Extract latitude and longitude
    neighborhoods["latitude"] = neighborhoods.geometry.y
    neighborhoods["longitude"] = neighborhoods.geometry.x

    stations["latitude"] = stations.geometry.y
    stations["longitude"] = stations.geometry.x

    # Assign unique neighborhood IDs
    neighborhoods["id"] = range(len(neighborhoods))

    # Assign unique station IDs
    stations["station_id"] = range(len(stations))

    print("\nCONNECTING LOCATIONS TO ROAD NETWORK")
    print("-" * 40)

    # Find nearest road node for every neighborhood
    neighborhood_nodes, neighborhood_distances = (
        ox.distance.nearest_nodes(
            graph,
            X=neighborhoods["longitude"].to_numpy(),
            Y=neighborhoods["latitude"].to_numpy(),
            return_dist=True
        )
    )

    neighborhoods["road_node"] = neighborhood_nodes
    neighborhoods["snap_distance_m"] = neighborhood_distances

    # Find nearest road node for every EMS station
    station_nodes, station_distances = (
        ox.distance.nearest_nodes(
            graph,
            X=stations["longitude"].to_numpy(),
            Y=stations["latitude"].to_numpy(),
            return_dist=True
        )
    )

    stations["road_node"] = station_nodes
    stations["snap_distance_m"] = station_distances

    print("Neighborhoods connected:", len(neighborhoods))
    print("EMS stations connected:", len(stations))

    print("Unique neighborhood road nodes:",
          neighborhoods["road_node"].nunique())

    print("Unique EMS road nodes:",
          stations["road_node"].nunique())

    print("\nSNAPPING DISTANCES")
    print("-" * 40)

    print("Average neighborhood distance (m):",
          round(neighborhoods["snap_distance_m"].mean(), 2))

    print("Maximum neighborhood distance (m):",
          round(neighborhoods["snap_distance_m"].max(), 2))

    print("Average EMS station distance (m):",
          round(stations["snap_distance_m"].mean(), 2))

    print("Maximum EMS station distance (m):",
          round(stations["snap_distance_m"].max(), 2))

    print("\nROAD NETWORK CONNECTIVITY")
    print("-" * 40)

    print("Weakly connected components:",
          nx.number_weakly_connected_components(graph))

    print("Strongly connected components:",
          nx.number_strongly_connected_components(graph))

    print("\nDETAILED CONNECTIVITY ANALYSIS")
    print("-" * 40)

    weak_components = sorted(
        nx.weakly_connected_components(graph),
        key=len,
        reverse=True
    )

    strong_components = sorted(
        nx.strongly_connected_components(graph),
        key=len,
        reverse=True
    )

    largest_weak = weak_components[0]
    largest_strong = strong_components[0]

    print("Largest weak component:",
          len(largest_weak))

    print("Largest strong component:",
          len(largest_strong))

    print("Percentage in largest strong component:",
          round(
              len(largest_strong) /
              graph.number_of_nodes() * 100, 2
          ))

    print("\nNeighborhoods in largest weak component:",
          neighborhoods["road_node"].isin(
              largest_weak
          ).sum())

    print("EMS stations in largest weak component:",
          stations["road_node"].isin(
              largest_weak
          ).sum())

    print("\nNeighborhoods in largest strong component:",
          neighborhoods["road_node"].isin(
              largest_strong
          ).sum())

    print("EMS stations in largest strong component:",
          stations["road_node"].isin(
              largest_strong
          ).sum())

    print("\nNeighborhoods outside largest strong component:")

    print(
        neighborhoods[
            ~neighborhoods["road_node"].isin(largest_strong)
        ][["id", "GEOID", "population", "road_node"]]
        .to_string(index=False)
    )

    print("\nEMS stations outside largest strong component:")

    print(
        stations[
            ~stations["road_node"].isin(largest_strong)
        ][["station_id", "road_node"]]
        .to_string(index=False)
    )


    # Restrict routing to the largest strongly connected component

    print("\nCONNECTIVITY CORRECTION")
    print("-" * 40)

    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()

    graph = graph.subgraph(largest_strong).copy()

    # Identify neighborhoods requiring reassignment
    affected = ~neighborhoods["road_node"].isin(graph.nodes)

    print("Neighborhoods requiring reassignment:",
          affected.sum())

    print("Affected population:",
          int(neighborhoods.loc[affected, "population"].sum()))

    # Preserve original snapping information
    neighborhoods["original_road_node"] = (
        neighborhoods["road_node"]
    )

    neighborhoods["original_snap_distance_m"] = (
        neighborhoods["snap_distance_m"]
    )

    # Reassign affected neighborhoods
    if affected.any():

        new_nodes, new_distances = ox.distance.nearest_nodes(
            graph,
            X=neighborhoods.loc[
                affected, "longitude"
            ].to_numpy(),
            Y=neighborhoods.loc[
                affected, "latitude"
            ].to_numpy(),
            return_dist=True
        )

        neighborhoods.loc[
            affected, "road_node"
        ] = new_nodes

        neighborhoods.loc[
            affected, "snap_distance_m"
        ] = new_distances

    print("\nREASSIGNED NEIGHBORHOODS")
    print("-" * 40)

    print(
        neighborhoods.loc[
            affected,
            [
                "id",
                "population",
                "original_road_node",
                "road_node",
                "original_snap_distance_m",
                "snap_distance_m"
            ]
        ].to_string(index=False)
    )

    # Verify all locations remain connected
    assert neighborhoods["road_node"].isin(
        graph.nodes
    ).all()

    assert stations["road_node"].isin(
        graph.nodes
    ).all()

    assert nx.is_strongly_connected(graph)

    print("\nFINAL CONNECTIVITY RESULTS")
    print("-" * 40)

    print("Original road nodes:", original_node_count)
    print("Retained road nodes:", graph.number_of_nodes())

    print("Removed road nodes:",
          original_node_count - graph.number_of_nodes())

    print("Original road edges:", original_edge_count)
    print("Retained road edges:", graph.number_of_edges())

    print("Strongly connected components:",
          nx.number_strongly_connected_components(graph))

    print("Connected neighborhoods:",
          neighborhoods["road_node"].isin(graph.nodes).sum())

    print("Connected EMS stations:",
          stations["road_node"].isin(graph.nodes).sum())

    print("Total population:",
          int(neighborhoods["population"].sum()))

    print("Average snapping distance (m):",
          round(neighborhoods["snap_distance_m"].mean(), 2))

    print("Maximum snapping distance (m):",
          round(neighborhoods["snap_distance_m"].max(), 2))

    print("\nConnectivity correction successful.")


    print("\nFIRST FIVE NEIGHBORHOODS")
    print("-" * 40)

    print(neighborhoods[
        [
            "id",
            "GEOID",
            "population",
            "latitude",
            "longitude",
            "vulnerability",
            "road_node"
        ]
    ].head().to_string(index=False))

    print("\nFIRST FIVE EMS STATIONS")
    print("-" * 40)

    print(stations[
        [
            "station_id",
            
            "latitude",
            "longitude",
            "road_node"
        ]
    ].head().to_string(index=False))

    return neighborhoods, stations, graph



def classify_neighborhoods(neighborhoods):

    print("\nPOPULATION DENSITY CLASSIFICATION")
    print("=" * 50)

    neighborhoods = neighborhoods.copy()

    # Verify land area is available
    if "ALAND" not in neighborhoods.columns:
        raise ValueError(
            "ALAND column missing from Census data."
        )

    # Convert square meters to square kilometers
    neighborhoods["land_area_km2"] = (
        pd.to_numeric(
            neighborhoods["ALAND"],
            errors="coerce"
        ) / 1_000_000
    )

    if (
        neighborhoods["land_area_km2"].isna().any()
        or (neighborhoods["land_area_km2"] <= 0).any()
    ):
        raise ValueError(
            "Invalid land area detected."
        )

    # Calculate population density
    neighborhoods["population_density"] = (
        neighborhoods["population"] /
        neighborhoods["land_area_km2"]
    )

    # Classify neighborhoods
    def classify_area(density):

        if density >= 1500:
            return "urban"

        elif density >= 300:
            return "suburban"

        else:
            return "rural"

    neighborhoods["area_type"] = (
        neighborhoods["population_density"]
        .apply(classify_area)
    )

    print("\nNEIGHBORHOOD DISTRIBUTION")
    print("-" * 40)

    print(
        neighborhoods["area_type"].value_counts()
    )

    print("\nPOPULATION BY AREA TYPE")
    print("-" * 40)

    print(
        neighborhoods.groupby("area_type")[
            "population"
        ].sum()
    )

    print("\nAVERAGE POPULATION DENSITY")
    print("-" * 40)

    print(
        neighborhoods.groupby("area_type")[
            "population_density"
        ].mean().round(2)
    )

    print("\nFIRST FIVE CLASSIFIED NEIGHBORHOODS")
    print("-" * 40)

    print(
        neighborhoods[
            [
                "id",
                "population",
                "land_area_km2",
                "population_density",
                "area_type"
            ]
        ].head().to_string(index=False)
    )

    return neighborhoods


def select_candidates(neighborhoods, seed=42):

    print("\nCANDIDATE STATION SELECTION")
    print("=" * 50)

    neighborhoods = neighborhoods.copy()

    neighborhoods["is_candidate"] = False

    total_candidates = round(
        len(neighborhoods) * 0.25
    )

    # Allocate candidates proportionally
    area_counts = neighborhoods[
        "area_type"
    ].value_counts()

    quotas = (
        area_counts / len(neighborhoods)
    ) * total_candidates

    allocation = quotas.astype(int)

    # Distribute remaining candidate slots
    remaining = (
        total_candidates - allocation.sum()
    )

    fractions = (
        quotas - allocation
    ).sort_values(ascending=False)

    for area in fractions.index[:remaining]:
        allocation[area] += 1

    # Select candidates within each area type
    for area, count in allocation.items():

        group = neighborhoods[
            neighborhoods["area_type"] == area
        ]

        selected = group.sample(
            n=int(count),
            random_state=seed
        )

        neighborhoods.loc[
            selected.index,
            "is_candidate"
        ] = True

    candidates = neighborhoods[
        neighborhoods["is_candidate"]
    ].copy()

    print("Total neighborhoods:",
          len(neighborhoods))

    print("Total candidate locations:",
          len(candidates))

    print("Candidate percentage:",
          round(
              len(candidates) /
              len(neighborhoods) * 100, 2
          ))

    print("\nCANDIDATES BY AREA TYPE")
    print("-" * 40)

    print(
        candidates["area_type"].value_counts()
    )

    assert len(candidates) == total_candidates

    return neighborhoods, candidates


if __name__ == "__main__":

    neighborhoods, stations, graph = load_county_data()

    neighborhoods = classify_neighborhoods(
        neighborhoods
    )

    neighborhoods, candidates = select_candidates(
        neighborhoods
    )

