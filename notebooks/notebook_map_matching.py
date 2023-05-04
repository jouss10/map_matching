from pathlib import Path

import osmnx as ox
import pandas as pd
import streamlit as st
from src.data.loaders import load_gps_points, load_osmnx_graph
from src.data.processing import (
    create_uuid_from_cols,
    get_road_segment_candidates,
    sample_points,
)
from src.model.build_hidden_markov_model import build_hidden_markov_model
from src.model.utils import add_shortest_path
from src.vizualisation.plots import plot_trip_matching


def main() -> None:
    st.title("Hidden Markov Map Matching")

    max_search_radius = 100
    nb_nearest_neighbors = 5
    sigma_z = 14
    beta = 200

    data_path = str(Path(__file__).parent.parent) + "/data/gps_points.zip"
    points = load_gps_points(data_path)
    trip_ids = points["trip_id"].unique()

    trip_id = st.selectbox(label="Choose a trip id", options=trip_ids)
    points = points[points["trip_id"] == trip_id]

    sampling_frequency = st.slider(
        "Sampling frequency of gps signals in seconds", 10, 600, 60
    )

    df = sample_points(points, sampling_frequency)
    # make copy of the geometry because in step 1 it will be used for a geopandas sjoin
    df["point_geometry"] = df["geometry"]

    osmnx_graph = load_osmnx_graph(points)
    nodes, edges = ox.graph_to_gdfs(osmnx_graph, fill_edge_geometry=True)
    edges["edges_geometry"] = edges["geometry"].tolist()

    # Step 1 : Road segment candidates selection
    candidates = get_road_segment_candidates(
        df, edges, max_search_radius, nb_nearest_neighbors
    )

    # Step 2 : Definition of HMM (states, observations, start_prob, emission_prob and transition_prob)
    # For each trip id, we will define a HMM model
    candidates = candidates.sort_values(["trip_id", "unixtime"])
    candidates["observation"] = candidates.groupby("trip_id")["unixtime"].rank(
        method="dense"
    )
    candidates = create_uuid_from_cols(
        candidates,
        uuid_column="states",
        cols=["edges_geometry", "observation"],
    )

    hmm_models = {}
    for trip_id in candidates["trip_id"].unique():
        df = candidates[candidates["trip_id"] == trip_id]
        hmm_models[trip_id] = build_hidden_markov_model(osmnx_graph, df, sigma_z, beta)

    # Step 3 : Best paths selection using Viterbi algorithm
    best_paths = {}
    for trip_id, hmm_model in hmm_models.items():
        best_paths[trip_id] = hmm_model.get_best_path()

    best_candidates = candidates.merge(
        pd.DataFrame(list(best_paths.items()), columns=["trip_id", "states"]).explode(
            "states"
        ),
        on=["trip_id", "states"],
    )

    # Step 4 : add shortest path
    # This step can be removed since we already compute the shortest path in the transition probabilities, but in order
    # to keep it, the viterbi algorithm should be updated. We preferred to keep the algorithm generic and add this part.
    result_df = pd.DataFrame()
    for trip_id in best_candidates["trip_id"].unique():
        df = best_candidates[best_candidates["trip_id"] == trip_id].sort_values(
            by="unixtime"
        )
        df = add_shortest_path(df, osmnx_graph, edges)
        df = df.assign(trip_id=trip_id)
        result_df = pd.concat([df, result_df], ignore_index=True)

    # Viz
    for trip_id in result_df["trip_id"].unique():
        st.markdown(f"### Most probable path for trip id {trip_id}")
        df = result_df[result_df["trip_id"] == trip_id]
        plot_trip_matching(df)


if __name__ == "__main__":
    main()
