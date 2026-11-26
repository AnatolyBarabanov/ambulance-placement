
import folium
import pandas as pd
import osmnx as ox
import geopandas as gpd
from pathlib import Path
from folium.plugins import Fullscreen

DATA_DIR = Path("data/processed")
OPT_DIR = DATA_DIR / "optimization"

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_map_data():

    neighborhoods = pd.read_csv(
        OPT_DIR / "neighborhood_accessibility.csv"
    )

    existing = pd.read_csv(
        DATA_DIR / "existing_stations.csv"
    )

    candidates = pd.read_csv(
        DATA_DIR / "candidate_stations.csv"
    )

    normal = pd.read_csv(
        OPT_DIR / "optimized_normal_stations.csv"
    )

    rush = pd.read_csv(
        OPT_DIR / "optimized_rush_stations.csv"
    )

    return neighborhoods, existing, candidates, normal, rush




def create_map(neighborhoods):

    county_map = folium.Map(
        location=[
            neighborhoods["latitude"].mean(),
            neighborhoods["longitude"].mean()
        ],
        zoom_start=11,
        tiles=None,
        control_scale=True,
        min_zoom=8,
        max_zoom=19
    )

    # Load the previously downloaded road network
    print("Loading Montgomery County roads...")

    graph = ox.load_graphml(
        "data/raw/montgomery_roads.graphml"
    )

    roads = ox.graph_to_gdfs(
        graph,
        nodes=False,
        edges=True
    )

    roads = roads.to_crs(epsg=4326)

    # Create road network layer
    road_layer = folium.FeatureGroup(
        name="Montgomery County Road Network",
        show=True
    )

    folium.GeoJson(
        roads[["geometry"]].to_json(),
        style_function=lambda feature: {
            "color": "#8c939c",
            "weight": 1,
            "opacity": 0.65
        },
        smooth_factor=1.0
    ).add_to(road_layer)

    road_layer.add_to(county_map)

    # Fit the map to Montgomery County
    county_map.fit_bounds(
        [
            [
                neighborhoods["latitude"].min(),
                neighborhoods["longitude"].min()
            ],
            [
                neighborhoods["latitude"].max(),
                neighborhoods["longitude"].max()
            ]
        ],
        padding=(30, 30),
        max_zoom=12
    )

    Fullscreen().add_to(county_map)

    # OpenStreetMap data attribution
    county_map.get_root().html.add_child(
        folium.Element(
            '<div style="position:fixed; bottom:0; '
            'right:0; z-index:9999; background:white; '
            'padding:4px; font-size:11px;">'
            '&copy; <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap contributors</a></div>'
        )
    )

    return county_map


def add_county_boundaries(county_map):

    print("Loading Census boundaries...")

    boundaries = gpd.read_file(
        "data/raw/pa_block_groups.zip"
    )

    boundaries = boundaries[
        (boundaries["STATEFP"] == "42") &
        (boundaries["COUNTYFP"] == "091")
    ].copy()

    boundaries = boundaries.to_crs(epsg=4326)

    layer = folium.FeatureGroup(
        name="Census Block Group Boundaries",
        show=True
    )

    folium.GeoJson(
        boundaries[["GEOID", "geometry"]].to_json(),
        style_function=lambda feature: {
            "color": "#64748b",
            "weight": 0.7,
            "fillColor": "#dbeafe",
            "fillOpacity": 0.08
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["GEOID"],
            aliases=["Census GEOID:"]
        )
    ).add_to(layer)

    layer.add_to(county_map)

    # Create the outer county boundary
    county_boundary = boundaries.dissolve()

    folium.GeoJson(
        county_boundary[["geometry"]].to_json(),
        name="Montgomery County Boundary",
        style_function=lambda feature: {
            "color": "#1e293b",
            "weight": 3,
            "fillOpacity": 0
        }
    ).add_to(county_map)


def add_station_layer(
    county_map,
    stations,
    layer_name,
    color,
    icon,
    show=True
):

    layer = folium.FeatureGroup(
        name=layer_name,
        show=show
    )

    for _, station in stations.iterrows():

        station_id = station.get(
            "station_id",
            station.get("id", "Unknown")
        )

        area = station.get(
            "area_type",
            "Not classified"
        )

        popup = f"""
        <b>{layer_name}</b><br>
        Station ID: {station_id}<br>
        Area type: {area}<br>
        Latitude: {station['latitude']:.6f}<br>
        Longitude: {station['longitude']:.6f}
        """

        folium.Marker(
            location=[
                station["latitude"],
                station["longitude"]
            ],
            popup=folium.Popup(
                popup,
                max_width=300
            ),
            tooltip=f"{layer_name}: {station_id}",
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="fa"
            )
        ).add_to(layer)

    layer.add_to(county_map)


def add_candidate_layer(county_map, candidates):

    layer = folium.FeatureGroup(
        name="All 148 candidate locations",
        show=False
    )

    for _, candidate in candidates.iterrows():

        folium.CircleMarker(
            location=[
                candidate["latitude"],
                candidate["longitude"]
            ],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"Candidate {candidate['id']}"
        ).add_to(layer)

    layer.add_to(county_map)


def add_coverage_layer(
    county_map,
    neighborhoods,
    column,
    layer_name,
    show=False
):

    layer = folium.FeatureGroup(
        name=layer_name,
        show=show
    )

    for _, neighborhood in neighborhoods.iterrows():

        travel_time = neighborhood[column]

        covered = travel_time <= 8

        color = "green" if covered else "red"

        status = (
            "Covered"
            if covered
            else "Outside 8-minute threshold"
        )

        popup = f"""
        <b>Neighborhood {neighborhood['id']}</b><br>
        GEOID: {neighborhood['GEOID']}<br>
        Area: {neighborhood['area_type']}<br>
        Population: {int(neighborhood['population']):,}<br>
        Vulnerability index: {neighborhood['vulnerability']:.3f}<br>
        Travel time: {travel_time:.2f} minutes<br>
        Status: {status}
        """

        folium.CircleMarker(
            location=[
                neighborhood["latitude"],
                neighborhood["longitude"]
            ],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(
                popup,
                max_width=350
            ),
            tooltip=(
                f"Neighborhood {neighborhood['id']} | "
                f"{travel_time:.2f} min"
            )
        ).add_to(layer)

    layer.add_to(county_map)


def add_area_classification(county_map, neighborhoods):

    layer = folium.FeatureGroup(
        name="Neighborhood area classification",
        show=False
    )

    colors = {
        "urban": "blue",
        "suburban": "orange",
        "rural": "purple"
    }

    for _, neighborhood in neighborhoods.iterrows():

        folium.CircleMarker(
            location=[
                neighborhood["latitude"],
                neighborhood["longitude"]
            ],
            radius=4,
            color=colors[neighborhood["area_type"]],
            fill=True,
            fill_opacity=0.7,
            tooltip=(
                f"Neighborhood {neighborhood['id']} | "
                f"{neighborhood['area_type']}"
            )
        ).add_to(layer)

    layer.add_to(county_map)


def add_bottleneck(county_map, neighborhoods):

    bottleneck = neighborhoods[
        neighborhoods["id"] == 418
    ].iloc[0]

    layer = folium.FeatureGroup(
        name="Geographic bottleneck",
        show=True
    )

    folium.Marker(
        location=[
            bottleneck["latitude"],
            bottleneck["longitude"]
        ],
        tooltip="Worst-served candidate neighborhood",
        popup=folium.Popup(
            """
            <b>Geographic Bottleneck</b><br>
            Neighborhood: 418<br>
            GEOID: 420912076004<br>
            Area: Rural<br>
            Population: 1,155<br>
            Normal lower bound: 9.4071 min<br>
            Rush lower bound: 12.4307 min
            """,
            max_width=350
        ),
        icon=folium.Icon(
            color="darkred",
            icon="exclamation-triangle",
            prefix="fa"
        )
    ).add_to(layer)

    layer.add_to(county_map)


def main():

    neighborhoods, existing, candidates, normal, rush = (
        load_map_data()
    )

    county_map = create_map(neighborhoods)
    add_county_boundaries(county_map)
    add_station_layer(
        county_map,
        existing,
        "Existing EMS stations",
        "blue",
        "ambulance",
        show=True
    )

    add_station_layer(
        county_map,
        normal,
        "Normal optimized stations",
        "green",
        "ambulance",
        show=False
    )

    add_station_layer(
        county_map,
        rush,
        "Rush optimized stations",
        "orange",
        "ambulance",
        show=False
    )

    add_candidate_layer(county_map, candidates)

    add_area_classification(
        county_map,
        neighborhoods
    )

    coverage_layers = [
        (
            "existing_ems_normal_time",
            "Existing EMS - Normal coverage"
        ),
        (
            "existing_ems_rush_time",
            "Existing EMS - Rush coverage"
        ),
        (
            "normal_optimized_normal_time",
            "Normal optimized - Normal coverage"
        ),
        (
            "normal_optimized_rush_time",
            "Normal optimized - Rush coverage"
        ),
        (
            "rush_optimized_normal_time",
            "Rush optimized - Normal coverage"
        ),
        (
            "rush_optimized_rush_time",
            "Rush optimized - Rush coverage"
        )
    ]

    for column, layer_name in coverage_layers:

        add_coverage_layer(
            county_map,
            neighborhoods,
            column,
            layer_name,
            show=False
        )

    add_bottleneck(county_map, neighborhoods)

    folium.LayerControl(
        collapsed=False
    ).add_to(county_map)

    output_file = (
        OUTPUT_DIR / "montgomery_county_map.html"
    )

    county_map.save(str(output_file))

    print("\nMAP GENERATION SUCCESSFUL")
    print("=" * 50)

    print("Neighborhoods:", len(neighborhoods))
    print("Existing EMS stations:", len(existing))
    print("Candidate locations:", len(candidates))
    print("Normal optimized stations:", len(normal))
    print("Rush optimized stations:", len(rush))

    print("Map saved to:", output_file)


if __name__ == "__main__":

    main()
