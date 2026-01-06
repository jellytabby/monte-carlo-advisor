import argparse
import logging
import os
from datetime import datetime
from typing import Set

import psutil
from matplotlib.pyplot import plot

import plot_main
import utils
from advisors.inline import inline_mc_advisor
from advisors.loop_unroll import loop_unroll_mc_advisor
from advisors.merged.merged_mc_advisor import MergedMonteCarloAdvisor
from datastructures import AdaptiveBenchmarkingResult

logger = logging.getLogger(__name__)
datefmt = "%Y-%m-%d %H:%M:%S"
fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(funcName)s(): %(message)s"


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        prog="Monte Carlo Autotuner",
        description="This python programs tunes compiler passes based on a Monte Carlo tree",
    )
    advisor_selection = parser.add_argument_group("Advisor Selection")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input. The script expects two files, <input>_main.[c|cpp] and <input>_kernel.[c|cpp] respectively.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Set the logging level to debug",
    )
    advisor_selection.add_argument(
        "-ia",
        "--inline-advisor",
        default=False,
        action="store_true",
        help="Enable the Inline Monte Carlo Advisor",
    )
    advisor_selection.add_argument(
        "-lua",
        "--loop-unroll-advisor",
        default=False,
        action="store_true",
        help="Enable the Loop Unroll Monte Carlo Advisor",
    )
    parser.add_argument(
        "-r",
        "--number-of-runs",
        type=int,
        default=50,
        help="Number of iterations to run the Monte Carlo Simulation",
    )
    parser.add_argument(
        "--min-run",
        default=False,
        action="store_true",
        help="Use the minimum runtime instead of the adaptive median",
    )
    parser.add_argument(
        "-w",
        "--warmup_runs",
        type=int,
        default=15,
        help="Number of warumup runs to discard before benchmarking.",
    )
    parser.add_argument(
        "-i",
        "--initial-samples",
        type=int,
        default=10,
        help="Number of initial samples the adaptive benchmark generates.",
    )
    parser.add_argument(
        "-m",
        "--max-samples",
        default=100,
        type=int,
        help="Maximum number of runtime samples to take until convergence.",
    )
    parser.add_argument(
        "-c",
        "--core",
        type=utils.comma_separated_numbers,
        required=True,
        help="Physical cores on which to execute the benchmark runs on.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="Timeout for llc and opt processes",
    )
    parser.add_argument(
        "--loop-unroll-advisor-model",
        type=str,
        help="Model to use to guide the unroll advisor",
    )
    parser.add_argument(
        "--plot-directory",
        type=str,
        default="plots",
        help="Directory to store generated plots and logs in.",
    )

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    MANAGER_PHYSICAL_CORES = 8
    physical_to_logical, _ = utils.get_core_maps()

    logical_cores = sum(
        [physical_to_logical[i] for i in range(MANAGER_PHYSICAL_CORES)], []
    )
    logger.info(f"Python script running on logical cores: {logical_cores}")
    os.sched_setaffinity(0, set(logical_cores))

    # next_free_core = physical_to_logical[
    #     utils.get_next_free_physical_core(MANAGER_PHYSICAL_CORES)
    # ][0]

    if len([c for c in args.core if c < MANAGER_PHYSICAL_CORES]) > 0:
        raise RuntimeError(f"Core {args.core} is reserved for the manager process")
    # try:
    #     next_free_core = physical_to_logical[args.core][0]
    # except:
    #     raise RuntimeError(f"Core {args.core} is out of range")

    benchmark_cores = sum([physical_to_logical[i] for i in args.core], [])

    logger.info(f"Benchmark core is {benchmark_cores}")
    logger.info(f"Script started with arguments: {args}")
    input_file = args.input_file
    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    os.environ["INPUT"] = input_file

    match (args.inline_advisor, args.loop_unroll_advisor):
        case (True, True):
            advisor = MergedMonteCarloAdvisor(
                input_name,
                unroll_model_path=args.loop_unroll_advisor_model,
            )
        case (True, False):
            advisor = inline_mc_advisor.InlineMonteCarloAdvisor(input_name)
        case (False, True):
            advisor = loop_unroll_mc_advisor.LoopUnrollMonteCarloAdvisor(
                input_name, model_path=args.loop_unroll_advisor_model
            )
        case _:
            raise Exception(
                "You need to specify at least one advisor. See '--help' for more information."
            )

    start = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter = plot_main.Plotter(input_name, args, advisor, start)
    make_clean()
    get_input_module()

    logger.info("Starting baseline benchmarking")
    baseline = get_baseline_runtime(
        args.warmup_runs,
        args.initial_samples,
        args.max_samples,
        set(benchmark_cores),
        args.min_run,
        plotter,
    )
    logger.info("Completed baseline benchmarking")

    logger.info("Starting Monte Carlo Tree runs")
    if args.min_run:
        assert baseline is list[float]
        scoring_function = lambda: get_min_score(
            baseline,
            args.warmup_runs,
            args.initial_samples,
            args.timeout,
            set(benchmark_cores),
            plotter,
        )
    else:
        assert type(baseline) is AdaptiveBenchmarkingResult
        scoring_function = lambda: get_median_score(
            baseline,
            args.warmup_runs,
            args.initial_samples,
            args.max_samples,
            args.timeout,
            set(benchmark_cores),
            plotter,
        )

    advisor.run_monte_carlo(
        args.number_of_runs, input_dir + "/", args.timeout, scoring_function
    )
    plotter.log_results()
    plotter.plot_speedup()
    del os.environ["INPUT"]  # NOTE: makes no difference apparently?
    logger.info("Succesfully completed Monte Carlo Advising")


def make_clean():
    cmd = ["make", "clean"]
    utils.get_cmd_output(cmd)


def get_input_module():
    cmd = ["make", "mod-pre-mc.bc"]
    utils.get_cmd_output(cmd)


def runtime_generator(cmd: list[str], cores: set[int]):
    logger.debug(cmd)
    while True:
        outs = utils.get_cmd_output(
            cmd, pre_exec_function=lambda: os.sched_setaffinity(0, cores)
        )
        yield utils.readout_mc_inline_timer(outs.decode())


def get_baseline_runtime(
    warmup_runs: int,
    initial_samples: int,
    max_samples: int,
    cores: set[int],
    use_min_run: bool,
    plotter: plot_main.Plotter,
) -> AdaptiveBenchmarkingResult | list[int]:

    cmd = ["make", "run_baseline"]
    if use_min_run:
        baseline_runtimes = utils.get_fixed_run_benchmark(
            runtime_generator(cmd, cores), warmup_runs, initial_samples
        )
        plotter.runtime_histogram(baseline_runtimes)
    else:
        baseline_runtimes = utils.adaptive_benchmark(
            runtime_generator(cmd, cores),
            warmup_runs=warmup_runs,
            initial_samples=initial_samples,
            max_samples=max_samples,
        )
        plotter.runtime_histogram(list(baseline_runtimes.runtimes))
    return baseline_runtimes


def get_median_score(
    baseline: utils.AdaptiveBenchmarkingResult,
    warmup_runs: int,
    initial_samples: int,
    max_samples: int,
    timeout: float,
    cores: set[int],
    plotter: plot_main.Plotter,
):
    cmd = ["make", "module_obj"]
    utils.get_cmd_output(cmd, timeout=timeout)
    cmd = ["make", "run"]
    runtimes = utils.adaptive_benchmark(
        runtime_generator(cmd, cores),
        warmup_runs=warmup_runs,
        initial_samples=initial_samples,
        max_samples=max_samples,
    )
    plotter.runtime_histogram(list(runtimes.runtimes))
    return baseline.median / runtimes.median
    # return utils.get_speedup_factor(baseline, runtimes)


def get_min_score(
    baseline: list[int],
    warmup_runs: int,
    initial_samples: int,
    timeout: float,
    cores: set[int],
    plotter: plot_main.Plotter,
) -> float:
    cmd = ["make", "module_obj"]
    utils.get_cmd_output(cmd, timeout=timeout)
    cmd = ["make", "run"]

    runtimes = utils.get_fixed_run_benchmark(
        runtime_generator(cmd, cores),
        warmup_runs=warmup_runs,
        initial_samples=initial_samples,
    )
    assert len(baseline) == len(runtimes)
    plotter.runtime_histogram(runtimes)
    return min(baseline) / min(runtimes)


if __name__ == "__main__":
    parse_args_and_run()
