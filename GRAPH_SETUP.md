# Graph Algorithms and County Data — Setup and Execution Guide

## 1. Overview

This branch contains the graph algorithms, synthetic county generator, real Montgomery County geographic data, ambulance station optimization, visualizations, automated tests and runtime experiments.

All required raw and processed datasets are included in the repository. No additional dataset downloads or Census API keys are required to use the existing data.

## 2. Installation

### Requirements

- Python 3.11 or newer
- Git
- Internet connection for installing Python dependencies
- Web browser for viewing interactive maps

Development and testing were performed using Python 3.13.14 on Windows.




### Install dependencies

```bash
python -m pip install -r requirements.txt
```

Run all commands below from the root of the repository.

## 3. Synthetic County and Graph Algorithms

### Generate the synthetic county

```bash
python -m backend.graph.generator
```

This generates `data/sample_map_500.json` using seed 42.

Expected output:

| Metric | Value |
|---|---:|
| Neighborhoods | 500 |
| Roads | 959 |
| Population | 1,159,405 |
| Candidate stations | 125 |
| Urban neighborhoods | 250 |
| Suburban neighborhoods | 175 |
| Rural neighborhoods | 75 |

### Run BFS and DFS

```bash
python -m backend.graph.traversal
```

This performs graph connectivity analysis using Breadth-First Search and Depth-First Search.

### Run Dijkstra

```bash
python -m backend.graph.dijkstra
```

This calculates shortest travel times and travel time matrices under normal and rush-hour traffic conditions.

### Run greedy p-center

```bash
python -m backend.graph.pcenter
```

This selects ambulance station locations while attempting to minimize the maximum travel time to any neighborhood.

## 4. Synthetic County Visualizations

The generated images are already included in the repository:

- `data/sample_map_500.png`
- `data/pcenter_20_normal.png`

To regenerate the synthetic county visualization:

```bash
python -m experiments.visualize_map
```

To regenerate the optimized station placement visualization:

```bash
python -m experiments.visualize_placement
```

## 5. Real Montgomery County, Pennsylvania

The real county implementation uses OpenStreetMap road data, Census geographic and demographic data, and Montgomery County EMS station records.

### Dataset Summary

| Metric | Value |
|---|---:|
| Populated neighborhoods | 594 |
| Total population | 867,573 |
| Existing EMS locations | 40 |
| Candidate station locations | 148 |
| Road network vertices | 26,376 |
| Directed road edges | 69,682 |
| Urban neighborhoods | 228 |
| Suburban neighborhoods | 304 |
| Rural neighborhoods | 62 |

### Quick Execution

All processed data and precomputed travel time matrices are already included.

Run the real county optimization:

```bash
python -m data.real_optimization
```

Run the optimization analysis:

```bash
python -m data.analyze_optimization
```

These commands do not require downloading the road network again.

### Reproduce the Complete Processing Pipeline

If you want to regenerate the processed datasets from the included raw files, run the following commands in order:

```bash
python -m data.validate_demographics
python -m data.real_county
python -m data.real_travel_times
python -m data.real_optimization
python -m data.analyze_optimization
```

The original data download scripts are also included, but they are unnecessary unless you want to refresh the source datasets.

## 6. Interactive Montgomery County Map

Generate the interactive geographic map:

```bash
python -m data.real_map
```

The generated file will be located at:

```text
visualizations/montgomery_county_map.html
```

Open the HTML file in a web browser.

Windows PowerShell:

```powershell
Start-Process "visualizations/montgomery_county_map.html"
```

The map includes:

- Montgomery County geographic boundaries
- Actual road network
- Neighborhood locations
- Existing EMS station locations
- Candidate ambulance station locations
- Optimized station placements
- Normal and rush-hour accessibility
- Urban, suburban and rural classifications
- Neighborhood accessibility and coverage layers

Use the layer control to toggle individual map layers.

The interactive HTML file is generated locally rather than stored on GitHub. All datasets necessary to generate it are included.

This is a standalone geographic visualization, separate from the final team website. Internet access may be required to load the map's JavaScript and CSS resources.

## 7. Automated Testing

### Synthetic Graph Tests

```bash
python -m pytest tests/test_graph.py -v
```

Expected result:

```text
28 passed
```

### Real County Tests

```bash
python -m pytest tests/test_real_county.py -v
```

Expected result:

```text
79 passed
```

### Complete Test Suite

```bash
python -m pytest tests -v
```

Most recent verified result:

```text
107 passed in 13.32s
```

Execution times may vary depending on the computer.

The tests cover graph generation, connectivity, shortest paths, station placement, real geographic data, travel time matrices, coverage calculations and optimization results.

## 8. Runtime Experiments

Run the runtime experiments:

```bash
python -m experiments.runtime
```

The experiments evaluate algorithm performance using synthetic graphs containing:

- 50 neighborhoods
- 100 neighborhoods
- 200 neighborhoods
- 400 neighborhoods
- 800 neighborhoods

Existing results and charts are located in:

```text
experiments/results/
```

Important files include:

| File | Description |
|---|---|
| runtime_results.csv | Measured execution times |
| runtime_comparison.png | Overall runtime comparison |
| bfs_runtime.png | BFS performance |
| dfs_runtime.png | DFS performance |
| dijkstra_runtime.png | Dijkstra performance |
| greedy_p_center_runtime.png | Greedy p-center performance |
| map_generator_runtime.png | Map generation performance |
| travel_time_matrix_runtime.png | Travel time matrix performance |
| final_test_results.txt | Saved automated test results |

Current runtime experiments cover the graph algorithms. Additional team algorithms will be benchmarked once their implementations are available.

## 9. Data Available for Application Integration

### Synthetic Map

```text
data/sample_map_500.json
```

The synthetic map supports both the shared team field names and the original graph implementation field names.

| Shared field | Original field |
|---|---|
| area | area_type |
| u | source |
| v | target |
| minutes | normal_time |
| rush_minutes | rush_hour_time |

Both naming conventions contain identical values.

### Real County Processed Data

All files are located in `data/processed/`.

| File | Description |
|---|---|
| neighborhoods.csv | Neighborhood coordinates, population and vulnerability |
| candidate_stations.csv | Potential ambulance station locations |
| existing_stations.csv | Existing EMS station locations |
| candidate_normal.npy | Candidate station normal travel times |
| candidate_rush.npy | Candidate station rush-hour travel times |
| existing_normal.npy | Existing EMS normal travel times |
| existing_rush.npy | Existing EMS rush-hour travel times |

CSV versions of all four travel time matrices are also included.

### Travel Time Matrix Dimensions

| Matrix | Dimensions |
|---|---|
| Candidate normal | 148 × 594 |
| Candidate rush | 148 × 594 |
| Existing normal | 40 × 594 |
| Existing rush | 40 × 594 |

Rows represent station locations. Columns represent neighborhoods. All travel times are measured in minutes.

Preserve the original row and column ordering when using the matrices.

The precomputed matrices allow placement algorithms to run without recalculating every shortest path.

The real county road network is directed and contains separate road vertices. It should not be treated as an undirected synthetic neighborhood graph.

### Optimization Results

All files are located in `data/processed/optimization/`.

| File | Description |
|---|---|
| placement_comparison.csv | Existing versus optimized station comparisons |
| area_type_comparison.csv | Urban, suburban and rural coverage |
| neighborhood_accessibility.csv | Neighborhood-level accessibility |
| optimized_normal_stations.csv | Optimized normal-traffic station locations |
| optimized_rush_stations.csv | Optimized rush-hour station locations |
| normal_optimization_history.csv | Normal-traffic optimization history |
| rush_optimization_history.csv | Rush-hour optimization history |

These datasets can be used directly for application charts, maps, coverage calculations and station placement comparisons.

## 10. Core Algorithm Functions

| File | Function |
|---|---|
| backend/graph/generator.py | generate_map(n, seed) |
| backend/graph/traversal.py | build_adjacency(county_map, rush_hour=False) |
| backend/graph/traversal.py | bfs_components(adj) |
| backend/graph/traversal.py | dfs_components(adj) |
| backend/graph/dijkstra.py | dijkstra(adj, source) |
| backend/graph/dijkstra.py | travel_time_matrix(adj, candidates) |
| backend/graph/pcenter.py | greedy_pcenter(matrix, p) |

These functions are available for importing into the team backend and placement algorithms.

## 11. Raw Datasets

All required raw datasets are included in `data/raw/`.

- montgomery_roads.graphml
- pa_block_groups.zip
- census_block_groups.geojson
- census_demographics.csv
- census_demographics_validated.csv
- ems_stations.geojson
- ems_stations_cleaned.geojson

A Census API key is only necessary if downloading updated Census demographic data through the original download script.

## 12. Data Sources and Limitations

Data sources:

- Montgomery County EMS geographic records
- OpenStreetMap
- United States Census Bureau TIGER/Line geographic data
- American Community Survey 2024 five-year estimates

Limitations:

- Rush-hour travel times are modeled rather than measured live.
- Road speeds are estimated from road network information.
- Candidate station locations are hypothetical.
- Existing EMS locations have not been independently verified for active staffing.
- Real county vulnerability scores use uninsured and elderly population proportions.
- Estimated travel times do not represent guaranteed emergency response times.

## 13. Recommended First Run

For teammates who want to quickly verify the contribution:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -v
python -m data.real_optimization
python -m data.real_map
```

Afterward, open:

```text
visualizations/montgomery_county_map.html
```

The graph algorithms, geographic datasets, processed travel time matrices and optimization results are then available for integration into the final team application.
