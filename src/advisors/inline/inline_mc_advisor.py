from __future__ import annotations

import logging
import random
from math import sqrt
from typing import Optional, final

from advisors.inline import inline_runner
from advisors.log_reader import TensorValue
from advisors.mc_advisor import MonteCarloAdvisor, State

logger = logging.getLogger(__name__)


@final
class InlineMonteCarloAdvisor(MonteCarloAdvisor[bool]):
    def __init__(self, input_name, path, timeout, C: float = sqrt(2)) -> None:
        super().__init__(input_name, path, timeout, C)
        self.runner = inline_runner.InlineCompilerCommunicator(input_name, True)
        self.filename = self.runner.channel_base

    def opt_args(self) -> list[str]:
        return [
            "opt",
            # "-passes=inline",
            "-O3",
            # "-passes=default<O3>,scc-oz-module-inliner",
            "-inliner-interactive-include-default",
            "-interactive-model-runner-echo-reply",
            "-debug-only=inline,inline-ml",
            "-enable-ml-inliner=release",
            f"-inliner-interactive-channel-base={self.filename}",
        ] + ["-o", self.path + "mod-post-mc.bc", self.path + "mod-pre-mc.bc"]

    def get_rollout_decision(self, tv=None, heuristic=None) -> bool:
        choice = random.random()
        return True if choice >= 0.5 else False

    def get_default_decision(
        self, advisor_type: str, tv, heuristic: Optional[bool]
    ) -> bool:
        return bool(tv[-1][0])

    def set_state_as_fully_explored(self, state: State[bool]):
        state.subtree_is_fully_explored = True
        current = state.parent
        while current:
            if len(current.children) == 2 and all(
                c.subtree_is_fully_explored for c in current.children
            ):
                current.subtree_is_fully_explored = True
                current = current.parent
            else:
                return

    def get_next_state(
        self,
        state: State[bool],
        tv: Optional[list[TensorValue]],
        heuristic=None,
    ) -> State[bool]:
        if state.is_leaf():
            choice = self.get_rollout_decision()
            return state.add_child(choice)
        elif len(state.children) == 2:
            return max(
                (
                    c for c in state.children if not c.subtree_is_fully_explored
                ),  # we only want unexplored paths
                key=self.uct,
            )
        elif state.children[0].decisions[-1] == False:
            return state.add_child(True)
        elif state.children[0].decisions[-1] == True:
            return state.add_child(False)
        else:
            assert False
