
"""
Problem:
Identify connected components within the county road network.

Input:
    adj: Adjacency list representing the road network.

Output:
    Connected components sorted from largest to smallest.

Design:
    BFS uses a FIFO queue.
    DFS uses a LIFO stack.
    Both maintain a visited set to avoid repeated visits.

Time Complexity:
    BFS and DFS traversal: O(V + E).
    Sorting C components: O(C log C).

Space Complexity:
    O(V).
"""

from collections import deque
import json


def build_adjacency(county_map, rush_hour=False):
    """Convert county map data into a weighted adjacency list."""

    adj = {
        node["id"]: []
        for node in county_map["neighborhoods"]
    }

    if rush_hour:
        weight = "rush_hour_time"
    else:
        weight = "normal_time"

    for road in county_map["roads"]:

        source = road["source"]
        target = road["target"]
        travel_time = road[weight]

        # Roads are undirected
        adj[source].append((target, travel_time))
        adj[target].append((source, travel_time))

    return adj


def bfs_components(adj):
    """Find connected components using BFS."""

    visited = set()
    components = []

    for start in adj:

        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)

        component = []

        while queue:

            current = queue.popleft()
            component.append(current)

            for neighbor, weight in adj[current]:

                if neighbor not in visited:

                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)

    return components


def dfs_components(adj):
    """Find connected components using iterative DFS."""

    visited = set()
    components = []

    for start in adj:

        if start in visited:
            continue

        stack = [start]
        visited.add(start)

        component = []

        while stack:

            current = stack.pop()
            component.append(current)

            for neighbor, weight in adj[current]:

                if neighbor not in visited:

                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)

    return components


if __name__ == "__main__":

    with open("data/sample_map_500.json", "r") as file:

        county_map = json.load(file)

    adj = build_adjacency(county_map)

    bfs_results = bfs_components(adj)
    dfs_results = dfs_components(adj)

    print("COUNTY CONNECTIVITY ANALYSIS")
    print("----------------------------")

    print("\nBFS Results")
    print("Connected components:", len(bfs_results))

    print(
        "Component sizes:",
        [len(c) for c in bfs_results]
    )

    print(
        "Largest component:",
        len(bfs_results[0])
    )

    print("\nDFS Results")
    print("Connected components:", len(dfs_results))

    print(
        "Component sizes:",
        [len(c) for c in dfs_results]
    )

    print(
        "Largest component:",
        len(dfs_results[0])
    )

    # Neighborhoods outside the largest component
    unreachable = set(adj) - set(bfs_results[0])

    print("\nUnreachable neighborhoods:", len(unreachable))
    print("Unreachable IDs:", sorted(unreachable))

    # Verify that BFS and DFS found identical components
    bfs_sets = {
        frozenset(c) for c in bfs_results
    }

    dfs_sets = {
        frozenset(c) for c in dfs_results
    }

    print(
        "\nBFS and DFS results match:",
        bfs_sets == dfs_sets
    )

    if len(bfs_results) == 1:

        print("Status: County is fully connected.")

    else:

        print("Status: County contains disconnected areas.")
