from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from ..algorithm.viterbi_algorithm import viterbi_algorithm


@dataclass
class HiddenMarkovModel:
    observations: Tuple[int]
    states: Tuple[str]
    start_prob: Dict[str, float]
    emission_prob: Dict[str, Dict[int, float]]
    transition_prob: Dict[str, Dict[str, float]]

    def get_best_path(self) -> List[str]:
        return viterbi_algorithm(
            self.observations,
            self.states,
            self.start_prob,
            self.emission_prob,
            self.transition_prob,
        )
