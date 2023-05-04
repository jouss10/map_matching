from math import asin, cos, exp, pi, radians, sin, sqrt

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def add_emission_probability(
    state_df: gpd.GeoDataFrame, sigma_z: float
) -> gpd.GeoDataFrame:
    state_df["distance_in_meters"] = state_df.apply(
        lambda row: haversine(
            row["point_geometry"].x,
            row["point_geometry"].y,
            row["projection_point"].x,
            row["projection_point"].y,
        ),
        axis=1,
    )

    state_df["emission_probability"] = state_df["distance_in_meters"].apply(
        lambda x: exp(-((x / sigma_z) ** 2) / 2) / (sqrt(2 * pi) * sigma_z)
    )
    return state_df


def add_transition_probability(
    state_df: gpd.GeoDataFrame,
    next_states_df: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    beta: float,
) -> gpd.GeoDataFrame:
    next_states_df = next_states_df.assign(
        previous_point_geometry=state_df["point_geometry"].values[0],
        previous_projection_point=state_df["projection_point"].values[0],
        previous_nearest_end_node=state_df["v"].values[0],
        previous_nearest_start_node=state_df["u"].values[0],
        previous_edges_geometry=state_df["edges_geometry"].values[0],
        previous_length=state_df["length"].values[0],
    )

    next_states_df["is_same_edges"] = np.where(
        (
            (next_states_df["previous_nearest_start_node"] == next_states_df["u"])
            & (next_states_df["previous_nearest_end_node"] == next_states_df["v"])
        ),
        1,
        0,
    )
    next_states_df["great_circle_distance_to_previous_point"] = next_states_df.apply(
        lambda row: haversine(
            row["previous_point_geometry"].x,
            row["previous_point_geometry"].y,
            row["point_geometry"].x,
            row["point_geometry"].y,
        ),
        axis=1,
    )

    next_states_df["route_distance_to_previous_node"] = next_states_df.apply(
        lambda row: driving_distance(graph, row["previous_nearest_end_node"], row["u"])
        if not row["is_same_edges"]
        else -row["previous_length"],
        axis=1,
    )
    next_states_df["route_distance_to_previous_point"] = (
        next_states_df["previous_length"]
        - next_states_df.apply(
            lambda row: row["previous_edges_geometry"].project(
                row["previous_projection_point"], normalized=True
            )
            * row["previous_length"],
            axis=1,
        )
        + next_states_df["route_distance_to_previous_node"]
        + next_states_df.apply(
            lambda row: row["edges_geometry"].project(
                row["projection_point"], normalized=True
            )
            * row["length"],
            axis=1,
        )
    )

    next_states_df["transition_probability"] = (
        np.exp(
            -abs(
                next_states_df["route_distance_to_previous_point"]
                - next_states_df["great_circle_distance_to_previous_point"]
            )
            / beta
        )
        / beta
    )
    return next_states_df


def add_shortest_path(
    df: gpd.GeoDataFrame, graph: nx.MultiDiGraph, edges: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    for obs in df["observation"].unique()[:-1]:
        next_obs = obs + 1

        u, v, time = (
            df.loc[df["observation"] == obs, "u"].values[0],
            df.loc[df["observation"] == obs, "v"].values[0],
            df.loc[df["observation"] == obs, "unixtime"].values[0],
        )
        next_u, next_v, next_time = (
            df.loc[df["observation"] == next_obs, "u"].values[0],
            df.loc[df["observation"] == next_obs, "v"].values[0],
            df.loc[df["observation"] == next_obs, "unixtime"].values[0],
        )
        if len({u, v, next_u, next_v}) < 4:
            continue

        else:
            path = nx.shortest_path(graph, v, next_u, weight="length")

            shortest_path_nodes = pd.DataFrame(
                [[path[i], path[i + 1]] for i in range(len(path) - 1)],
                columns=["u", "v"],
            )

            shortest_path_nodes = shortest_path_nodes.assign(
                observation=(obs + next_obs) / 2, unixtime=time + (next_time - time) / 2
            )
            shortest_path_nodes = shortest_path_nodes.merge(
                edges, how="left", on=["u", "v"]
            )
            df = pd.concat([df, shortest_path_nodes])
    return df


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6.371e6  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def driving_distance(graph: nx.MultiDiGraph, orig_node: int, dest_node: int) -> float:
    try:
        distance = nx.shortest_path_length(graph, orig_node, dest_node, weight="length")
    except nx.exception.NetworkXNoPath:
        distance = 1e10

    return distance
