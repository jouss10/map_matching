from typing import Dict, List, Tuple, Union

import numpy as np


def viterbi_algorithm(
    observations: Tuple[int],
    states: Tuple[str],
    start_prob: Dict[str, float],
    emission_prob: Dict[str, Dict[int, float]],
    transition_prob: Dict[str, Dict[str, float]],
) -> List[str]:
    V = [{}]
    for state in states:
        V[0][state] = {
            "prob": start_prob[state] * emission_prob[state][observations[0]],
            "prev": None,
        }

    for i in range(1, len(observations)):
        V.append({})
        for state in states:
            V[i][state] = {
                "prob": max(
                    [
                        V[i - 1][s]["prob"]
                        * transition_prob[s][state]
                        * emission_prob[state][observations[i]]
                        for s in states
                    ]
                ),
                "prev": states[
                    np.argmax(
                        [
                            V[i - 1][s]["prob"]
                            * transition_prob[s][state]
                            * emission_prob[state][observations[i]]
                            for s in states
                        ]
                    )
                ],
            }

    max_ = 0
    best_path_pointer = ""
    for state in states:
        if V[len(observations) - 1][state]["prob"] > max_:
            max_ = V[len(observations) - 1][state]["prob"]
            best_path_pointer = state

    best_path = [best_path_pointer]
    previous = best_path_pointer
    for i in range(len(observations) - 2, -1, -1):
        best_path.insert(0, V[i + 1][previous]["prev"])
        previous = V[i + 1][previous]["prev"]

    return best_path
