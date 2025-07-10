import logging
from math import sqrt
from typing import Any, Optional

from typing_extensions import override

import utils
from advisors import log_reader
from advisors.inline.inline_mc_advisor import InlineMonteCarloAdvisor
from advisors.loop_unroll.loop_unroll_mc_advisor import LoopUnrollMonteCarloAdvisor
from advisors.mc_advisor import MonteCarloAdvisor, State
from advisors.merged.merged_runner import MergedCompilerCommunicator
from utils import INLINE, LOOP_UNROLL, MonteCarloError, UnknownAdvisorError

logger = logging.getLogger(__name__)


class MergedMonteCarloAdvisor(MonteCarloAdvisor[bool | int]):
    def __init__(self, input_name, path, timeout, unroll_model_path, C=sqrt(2)) -> None:
        super().__init__(input_name, path, timeout, C)
        self.inline_advisor = InlineMonteCarloAdvisor(input_name, path, timeout)
        self.loop_unroll_advisor = LoopUnrollMonteCarloAdvisor(
            input_name, path, timeout, unroll_model_path
        )

        self.runner = MergedCompilerCommunicator(input_name, True)

    def opt_args(self) -> list[str]:
        return [
            "opt",
            "-O3",
            # "-passes=default<O3>,loop-unroll",
            "-interactive-model-runner-echo-reply",
            "-inliner-interactive-include-default",
            # "-debug-only=inline,inline-ml",
            "-enable-ml-inliner=release",
            f"-inliner-interactive-channel-base={self.inline_advisor.filename}",
            f"--mlgo-loop-unroll-interactive-channel-base={self.loop_unroll_advisor.filename}",
            "--mlgo-loop-unroll-advisor-mode=development",
            "-debug-only=loop-unroll-development-advisor,loop-unroll,inline,inline-ml",
        ] + ["-o", self.path + "mod-post-mc.bc", self.path + "mod-pre-mc.bc"]

    def get_next_state(
        self,
        state: State[bool | int],
        tv: list[log_reader.TensorValue],
        heuristic: Optional[int],
        advisor_type: str = "",
    ) -> State:
        if state.is_leaf():
            choice = self.get_rollout_decision(tv, heuristic, advisor_type)
            return state.add_child(choice)
        assert (type(state.children[0].decisions[-1]) is bool) == (
            advisor_type == utils.INLINE
        )  # if we have an inlining decision, we expect the children to be inline == bool decision states
        match advisor_type:
            case utils.INLINE:
                return self.inline_advisor.get_next_state(state, tv)
            case utils.LOOP_UNROLL:
                return self.loop_unroll_advisor.get_next_state(state, tv, heuristic)
            case _:
                raise UnknownAdvisorError()

    def get_rollout_decision(
        self,
        tv: list[log_reader.TensorValue],
        heuristic: Optional[int],
        advisor_type: str = "",
    ) -> bool | int:
        match advisor_type:
            case utils.INLINE:
                return self.inline_advisor.get_rollout_decision()
            case utils.LOOP_UNROLL:
                return self.loop_unroll_advisor.get_rollout_decision(tv, heuristic)
            case _:
                raise UnknownAdvisorError()

    def get_default_decision(
        self, advisor_type: str, tv: list[log_reader.TensorValue], heuristic
    ) -> bool | int:
        match advisor_type:
            case utils.INLINE:
                return self.inline_advisor.get_default_decision(advisor_type, tv, None)
            case utils.LOOP_UNROLL:
                return self.loop_unroll_advisor.get_default_decision(
                    advisor_type, tv, heuristic
                )
            case _:
                raise UnknownAdvisorError()

    def set_state_as_fully_explored(self, state: State[int]):
        state.subtree_is_fully_explored = True
        current = state.parent
        while current:
            if type(current.children[0].decisions[-1]) is bool:
                all_children_visited = len(current.children) == 2
            elif type(current.children[0].decisions[-1]) is int:
                all_children_visited = (
                    len(current.children) == self.loop_unroll_advisor.MAX_UNROLL_FACTOR
                )
            else:
                logger.error("Decisions should only be int or bools... what happened?")
                exit(-1)
            if all_children_visited and all(
                c.subtree_is_fully_explored for c in current.children
            ):
                current.subtree_is_fully_explored = True
                current = current.parent
            else:
                return

    def check_unroll_success(self, action: bool):
        if action:
            return
        if self.in_rollout and self.current_path[-1] != 1:
            # we are in rollout, so we don't want to block off our mc state, but want the path to accurately reflect what choices we made
            self.current_path[-1] = 1
            logger.debug("Unsuccessful unrolling during rollout")
            return
        if not self.in_rollout and self.current and self.current.decisions[-1] != 1:
            # assert self.current.is_leaf()
            # should only happen in leaf states, otherwise would have triggered earlier
            logger.warning("Unsuccessful unrolling")
            raise MonteCarloError("unsuccessful unrolling")

    @override
    def advice(self, advisor_type: str, tv, heuristic) -> Any:
        assert self.current
        if self.current.visits == 0:
            self.in_rollout = True
            decision = self.get_rollout_decision(tv, heuristic, advisor_type)
        else:
            next = self.get_next_state(self.current, tv, heuristic, advisor_type)
            self.current = next
            decision = next.decisions[-1]
        self.current_path.append(decision)
        logger.debug(f"Current path: {self.current_path}")
        return self.wrap_advice(advisor_type, decision)

    @override
    def wrap_advice(self, advisor_type: str, advice: bool | int) -> bool | int:
        match advisor_type:
            case utils.INLINE:
                return advice
            case utils.LOOP_UNROLL:
                return self.loop_unroll_advisor.wrap_advice(advisor_type, advice)
            case _:
                raise UnknownAdvisorError()

    @override
    def get_score(self, scoring_function):

        logger.error(f"selfpath: {self.path}")
        self.runner.compile_once(
            self.opt_args(),
            self.advice,
            on_action=self.check_unroll_success,
            timeout=self.timeout,
        )
        return scoring_function()
