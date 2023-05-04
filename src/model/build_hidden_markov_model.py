import geopandas as gpd
import networkx as nx

from .hidden_markov_model import HiddenMarkovModel
from .utils import add_emission_probability, add_transition_probability


def build_hidden_markov_model(
    graph: nx.MultiDiGraph, df: gpd.GeoDataFrame, sigma_z: float, beta: float
) -> HiddenMarkovModel:
    df = add_emission_probability(df, sigma_z)
    df = df.sort_values(by="unixtime")

    observations = tuple(df["observation"].unique())
    states = tuple(df["states"].unique())

    start_prob = {}
    sub_df = df[df["observation"] == observations[0]]
    for state in states:
        if state in sub_df["states"].unique():
            start_prob[state] = sub_df[sub_df["states"] == state][
                "emission_probability"
            ].values[0]
        else:
            start_prob[state] = 0

    sum_start_prob = sum(start_prob.values())
    start_prob = {u: v / sum_start_prob for u, v in start_prob.items()}

    emission_prob = {
        s: dict(zip(observations, [0] * len(observations))) for s in states
    }
    for state in states:
        state_df = df[df["states"] == state]
        emission_prob[state].update(
            dict(
                zip(
                    list(state_df["observation"]),
                    list(state_df["emission_probability"]),
                )
            )
        )

    transition_prob = {s: dict(zip(states, [0] * len(states))) for s in states}
    for state in states:
        state_df = df[df["states"] == state]
        next_states_df = df[df["observation"] == state_df["observation"].values[0] + 1]
        if next_states_df.shape[0] == 0:
            continue

        next_states_df = add_transition_probability(
            state_df, next_states_df, graph, beta
        )

        transition_prob[state].update(
            dict(
                zip(
                    list(next_states_df["states"]),
                    list(next_states_df["transition_probability"]),
                )
            )
        )
    return HiddenMarkovModel(
        observations, states, start_prob, emission_prob, transition_prob
    )
