
"""
Montgomery County Data Collection

Purpose:
    Download real geographic data for Montgomery
    County, Pennsylvania.

Data sources:
    Montgomery County ArcGIS
    United States Census Bureau
    OpenStreetMap
"""

import requests
import json
from pathlib import Path


OUTPUT_DIR = Path("data/raw")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_NAME = "Montgomery County, Pennsylvania, USA"

STATE_FIPS = "42"
COUNTY_FIPS = "091"

EMS_URL = (
    "https://services1.arcgis.com/"
    "kOChldNuKsox8qZD/ArcGIS/rest/services/"
    "Montgomery_County_EMS_Station/FeatureServer"
)


def download_ems_stations():

    print("\nDOWNLOADING EMS STATIONS")
    print("-" * 40)

    response = requests.get(
        EMS_URL,
        params={"f": "json"},
        timeout=60
    )

    response.raise_for_status()

    service = response.json()

    if "error" in service:
        raise RuntimeError(service["error"])

    layers = service.get("layers", [])

    if not layers:
        raise RuntimeError("No EMS layers found.")

    print("Available layers:")

    for layer in layers:
        print(layer["id"], layer["name"])

    layer_id = layers[0]["id"]

    url = f"{EMS_URL}/{layer_id}/query"

    features = []
    offset = 0

    while True:

        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 100
        }

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        batch = data.get("features", [])

        if not batch:
            break

        features.extend(batch)

        offset += len(batch)

    if not features:
        raise RuntimeError("No EMS stations downloaded.")

    stations = {
        "type": "FeatureCollection",
        "features": features
    }

    filename = OUTPUT_DIR / "ems_stations.geojson"

    with open(filename, "w", encoding="utf-8") as file:

        json.dump(stations, file, indent=4)

    print("EMS records downloaded:", len(features))
    print("Saved to:", filename)

    print("\nAVAILABLE DATA FIELDS")
    print("-" * 40)

    for field in features[0]["properties"]:
        print(field)

    print("\nFIRST FIVE EMS RECORDS")
    print("-" * 40)

    for station in features[:5]:

        print(
            json.dumps(
                station["properties"],
                indent=2
            )
        )

        print("Coordinates:", station["geometry"])
        print()

    return stations


def download_census_block_groups():

    import geopandas as gpd

    print("\nDOWNLOADING CENSUS BLOCK GROUPS")
    print("-" * 40)

    url = (
        "https://www2.census.gov/geo/tiger/"
        "TIGER2024/BG/tl_2024_42_bg.zip"
    )

    zip_path = OUTPUT_DIR / "pa_block_groups.zip"

    if not zip_path.exists():

        response = requests.get(
            url,
            timeout=180
        )

        response.raise_for_status()

        zip_path.write_bytes(response.content)

        print("Pennsylvania boundaries downloaded.")

    else:

        print("Using previously downloaded boundaries.")

    
def download_census_demographics():

    import os
    import pandas as pd

    print("\nDOWNLOADING CENSUS DEMOGRAPHICS")
    print("-" * 40)

    url = "https://api.census.gov/data/2024/acs/acs5"

    api_key = os.environ.get("CENSUS_API_KEY")

    # Male and female residents aged 65+
    elderly_columns = [
        f"B01001_{i:03d}E"
        for i in list(range(20, 26)) +
        list(range(44, 50))
    ]

    variables = [
    "B01003_001E",
    "B01001_001E"
] + elderly_columns
    api_key = os.environ.get("CENSUS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "census api key is missing."
        )
    def fetch_data(columns):

        params = {
            "get": ",".join(columns),
            "for": "block group:*",
            "in": "state:42 county:091 tract:*"
        }

        if api_key:
            params["key"] = api_key

        response = requests.get(
            url,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        records = response.json()

        df = pd.DataFrame(
            records[1:],
            columns=records[0]
        )

        df["GEOID"] = (
            df["state"] +
            df["county"] +
            df["tract"] +
            df["block group"]
        )

        return df

    demographics = fetch_data(variables)

    # Retrieve insurance table definitions
    metadata_url = (
        url + "/groups/B27010.json"
    )

    response = requests.get(
        metadata_url,
        timeout=120
    )

    response.raise_for_status()

    metadata = response.json()["variables"]

    # Find estimates for residents without insurance
    uninsured_columns = [
        code
        for code, details in metadata.items()
        if code.endswith("E")
        and code.startswith("B27010_")
        and details.get("label", "").lower()
        .rstrip(":").endswith("no health insurance coverage")
    ]

    if not uninsured_columns:
        raise RuntimeError(
            "Uninsured Census variables not found."
        )

    print("Uninsured variables:", uninsured_columns)

    insurance = fetch_data(
        ["B27010_001E"] + uninsured_columns
    )

    df = demographics.merge(
        insurance[
            ["GEOID", "B27010_001E"] +
            uninsured_columns
        ],
        on="GEOID",
        validate="one_to_one"
    )

    numeric_columns = (
        variables +
        ["B27010_001E"] +
        uninsured_columns
    )

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Negative Census values indicate unavailable estimates
        df.loc[df[column] < 0, column] = float("nan")

    def calculate_rate(numerator, denominator):

        return numerator.div(
            denominator.where(denominator > 0)
        ).clip(0, 1)

    df["population"] = df["B01003_001E"]

    df["elderly_count"] = df[elderly_columns].sum(
        axis=1,
        min_count=len(elderly_columns)
    )

    df["elderly"] = calculate_rate(
        df["elderly_count"],
        df["B01001_001E"]
    )

    

    df["uninsured_count"] = df[uninsured_columns].sum(
        axis=1,
        min_count=len(uninsured_columns)
    )

    df["uninsured"] = calculate_rate(
        df["uninsured_count"],
        df["B27010_001E"]
    )

    df["vulnerability"] = df[
        ["uninsured", "elderly"]
    ].mean(axis=1, skipna=False)

    results = df[
        [
            "GEOID",
            "population",
            "uninsured",
            "elderly",
            "vulnerability"
        ]
    ].copy()

    filename = OUTPUT_DIR / "census_demographics.csv"

    results.to_csv(filename, index=False)

    print("Block groups:", len(results))
    print("Total population:", results["population"].sum())
    print("Average vulnerability:", results["vulnerability"].mean())

    print("\nMISSING VALUES")
    print(results.isnull().sum())

    print("\nFIRST FIVE NEIGHBORHOODS")
    print(results.head().to_string(index=False))

    print("\nSaved to:", filename)

    return results

    # Load Pennsylvania block groups
    block_groups = gpd.read_file(zip_path)

    # Filter Montgomery County
    county = block_groups[
        (block_groups["STATEFP"] == STATE_FIPS) &
        (block_groups["COUNTYFP"] == COUNTY_FIPS)
    ].copy()

    if county.empty:
        raise RuntimeError(
            "No Montgomery County block groups found."
        )

    # Calculate centroids using a projected CRS
    projected = county.to_crs("EPSG:26918")

    centroids = projected.geometry.centroid

    # Convert centroids back to latitude/longitude
    centroids = centroids.to_crs("EPSG:4326")

    county = county.to_crs("EPSG:4326")

    county["longitude"] = centroids.x
    county["latitude"] = centroids.y

    # Keep useful geographic fields
    county = county[
        [
            "GEOID",
            "STATEFP",
            "COUNTYFP",
            "TRACTCE",
            "BLKGRPCE",
            "ALAND",
            "AWATER",
            "latitude",
            "longitude",
            "geometry"
        ]
    ]

    filename = (
        OUTPUT_DIR / "census_block_groups.geojson"
    )

    county.to_file(
        filename,
        driver="GeoJSON"
    )

    print("County: Montgomery County, PA")
    print("Block groups:", len(county))
    print("Saved to:", filename)

    print("\nFIRST FIVE NEIGHBORHOODS")
    print("-" * 40)

    print(
        county[
            [
                "GEOID",
                "latitude",
                "longitude"
            ]
        ].head().to_string(index=False)
    )

    return county


def clean_ems_stations():

    import json

    print("\nCLEANING EMS STATIONS")
    print("-" * 40)

    input_file = OUTPUT_DIR / "ems_stations.geojson"
    output_file = OUTPUT_DIR / "ems_stations_cleaned.geojson"

    with open(input_file, encoding="utf-8") as file:
        data = json.load(file)

    locations = {}
    county_records = 0

    for feature in data["features"]:

        properties = feature["properties"]

        if properties.get("County") != "Montgomery County":
            continue

        county_records += 1

        coordinates = feature["geometry"]["coordinates"]

        key = tuple(coordinates)

        if key not in locations:

            locations[key] = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "station_id": len(locations),
                    "name": properties.get("DsplayName"),
                    "address": properties.get("Address"),
                    "city": properties.get("City"),
                    "municipality": properties.get("Municipality"),
                    "county": properties.get("County"),
                    "ems_numbers": [],
                    "agency_names": []
                }
            }

        station = locations[key]["properties"]

        ems_number = properties.get("EMS_Num")
        agency_name = properties.get("DsplayName")

        if ems_number not in station["ems_numbers"]:
            station["ems_numbers"].append(ems_number)

        if agency_name not in station["agency_names"]:
            station["agency_names"].append(agency_name)

    cleaned_data = {
        "type": "FeatureCollection",
        "features": list(locations.values())
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(cleaned_data, file, indent=4)

    print("Original records:", len(data["features"]))
    print("Montgomery County records:", county_records)
    print("Unique EMS locations:", len(locations))
    print("Duplicate locations:", county_records - len(locations))
    print("Saved to:", output_file)

    return cleaned_data


if __name__ == "__main__":

    print("MONTGOMERY COUNTY DATA COLLECTION")
    print("=" * 50)

    clean_ems_stations()
    download_census_demographics()