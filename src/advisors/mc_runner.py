import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import utils
from advisors import log_reader

logger = logging.getLogger(__name__)


class CompilerCommunicator(ABC):
    def __init__(self, input_name, debug):
        self.channel_base: str = input_name + type(self).__name__
        self.to_compiler = self.channel_base + ".in"
        self.from_compiler = self.channel_base + ".out"
        self.debug: bool = debug

    @abstractmethod
    def communicate_with_proc(
        self,
        compiler_proc: subprocess.Popen[bytes],
        advice: Callable[[str, list[log_reader.TensorValue], Optional[int]], int],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        timeout: Optional[float] = None,
    ): ...

    def set_nonblocking(self, pipe):
        os.set_blocking(pipe.fileno(), False)

    def set_blocking(self, pipe):
        os.set_blocking(pipe.fileno(), True)

    def clean_up_pipes(self):
        logger.debug(f"Cleaning up pipes for {type(self).__name__} ")
        try:
            os.unlink(self.to_compiler)
        except FileNotFoundError:
            pass
        try:
            os.unlink(self.from_compiler)
        except FileNotFoundError:
            pass

    def compile_once(
        self,
        process_and_args: list[str],
        advice: Callable[[str, list[log_reader.TensorValue], Optional[int]], int],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        timeout: Optional[float] = None,
    ):
        self.clean_up_pipes()
        with tempfile.TemporaryFile("b+x") as error_buffer:
            compiler_proc = None
            try:
                logger.debug(f"Launching compiler {' '.join(process_and_args)}")
                compiler_proc = subprocess.Popen(
                    process_and_args,
                    stderr=subprocess.DEVNULL if not self.debug else error_buffer,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )
                logger.debug("Sending module")
                self.communicate_with_proc(
                    compiler_proc, advice, on_features, on_heuristic, on_action, timeout
                )

            finally:
                assert compiler_proc
                if compiler_proc.returncode is None:
                    utils.terminate(compiler_proc)
                    self.clean_up_pipes()
                else:
                    status = utils.clean_up_process(compiler_proc, error_buffer)
                    self.clean_up_pipes()
                    if (
                        status != 0 and status != -2 and status != -15
                    ):  # -2 is Ctrl-C and -15 is us terminating the process
                        logger.error(f"Process failed with error code: {status}")
                        exit(status)
