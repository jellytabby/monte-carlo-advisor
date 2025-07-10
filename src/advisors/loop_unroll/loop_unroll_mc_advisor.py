import logging
import random
from math import sqrt
from typing import Optional, final

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from typing_extensions import override

from advisors.loop_unroll import loop_unroll_runner
from advisors.mc_advisor import MonteCarloAdvisor, State
from utils import LOOP_UNROLL, MonteCarloError

logger = logging.getLogger(__name__)


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(
        self,
        input_name: str,
        path: str,
        timeout: Optional[float],
        model_path: Optional[str] = None,
        C: float = sqrt(2),
    ) -> None:
        super().__init__(input_name, path, timeout, C)
        self.runner: loop_unroll_runner.LoopUnrollCompilerCommunicator = (
            loop_unroll_runner.LoopUnrollCompilerCommunicator(input_name, True)
        )
        self.filename = self.runner.channel_base

        self.MAX_UNROLL_FACTOR = 32

        if model_path is not None:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.infer_speedups = self.interpreter.get_signature_runner()
        else:
            self.interpreter = None
            self.infer_speedups = None

    def opt_args(self) -> list[str]:
        return [
            "opt",
            "-O3",
            # "-passes=default<O3>,loop-unroll",
            f"--mlgo-loop-unroll-interactive-channel-base={self.runner.channel_base}",
            "--mlgo-loop-unroll-advisor-mode=development",
            "--interactive-model-runner-echo-reply",
            "-debug-only=loop-unroll-development-advisor,loop-unroll",
        ] + ["-o", self.path + "mod-post-mc.bc", self.path + "mod-pre-mc.bc"]

    def make_response_for_factor(self, factor: int):
        assert factor >= -1
        return factor

    @override
    def wrap_advice(self, advisor_type: str, advice: int) -> int:
        return self.make_response_for_factor(advice)

    def has_model(self) -> bool:
        return self.interpreter is not None and self.infer_speedups is not None

    def get_model_predictions(self, tv) -> Optional[list[float]]:
        if not self.has_model():
            logger.debug(f"No model to use")
            return None
        input_dict = {
            t.spec().name: tf.expand_dims(tf.convert_to_tensor(t.to_numpy()), axis=0)
            for t in tv
        }
        res = self.infer_speedups(**input_dict)
        res = res["unrolling_decision"][0]
        logger.debug(f"Got unroll model output: {res}")
        return res

    def get_rollout_decision(self, tv, heuristic) -> int:
        model_prediction = self.get_model_predictions(tv)
        if model_prediction is not None and len(model_prediction) > 0:
            truncated_predictions = model_prediction[: self.MAX_UNROLL_FACTOR - 1]
            decision = int(np.argmax(truncated_predictions)) + 2
            decision_value = max(truncated_predictions)
            decision = (
                decision
                if decision_value >= 1.0
                else self.get_default_decision(LOOP_UNROLL, tv, heuristic)
            )
            logger.info(f"Model unrolling decision: {decision}")
            return decision
        return random.randint(1, self.MAX_UNROLL_FACTOR)

    def get_default_decision(
        self, advisor_type: str, tv, heuristic: Optional[int]
    ) -> int:
        assert heuristic is not None
        match heuristic:
            case -1:  # compiler returns -1 when no opinion
                return 1
            case 0:
                return 1  # ngl, not sure if this is even possible
            case heuristic if heuristic > self.MAX_UNROLL_FACTOR:
                return self.MAX_UNROLL_FACTOR
            case heuristic:
                return heuristic

    def set_state_as_fully_explored(self, state: State[int]):
        state.subtree_is_fully_explored = True
        current = state.parent
        while current:
            if len(current.children) == self.MAX_UNROLL_FACTOR and all(
                c.subtree_is_fully_explored for c in current.children
            ):
                current.subtree_is_fully_explored = True
                current = current.parent
            else:
                return

    def get_next_state(self, state: State[int], tv, heuristic) -> State[int]:
        # TODO do something with these
        if state.is_leaf():
            choice = self.get_rollout_decision(tv, heuristic)
            return state.add_child(choice)
        if len(state.children) == self.MAX_UNROLL_FACTOR:
            return max(
                (c for c in state.children if not c.subtree_is_fully_explored),
                key=self.uct,
            )
        else:
            visited_unroll_factors = set([c.decisions[-1] for c in state.children])
            remaining_unroll_factors = (
                set(range(1, self.MAX_UNROLL_FACTOR + 1)) - visited_unroll_factors
            )
            return state.add_child(random.choice(list(remaining_unroll_factors)))

    def check_unroll_success(self, action: bool):
        if action:
            return
        if self.in_rollout and self.current_path[-1] != 1:
            # we are in rollout, so we don't want to block off our mc state, but want the path to accurately reflect what choices we made
            self.current_path[-1] = 1
            return
        if not self.in_rollout and self.current and self.current.decisions[-1] != 1:
            # assert self.current.is_leaf()
            # should only happen in leaf states, otherwise would have triggered earlier
            logger.warning("Unsuccessful unrolling")
            raise MonteCarloError("unsuccessful unrolling")

    @override
    def get_score(self, scoring_function):
        self.runner.compile_once(
            self.opt_args(),
            self.advice,
            on_action=self.check_unroll_success,
            timeout=self.timeout,
        )
        return scoring_function()
