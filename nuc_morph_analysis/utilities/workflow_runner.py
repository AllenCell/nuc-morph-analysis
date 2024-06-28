# This module is more general than just nuc-morph-analysis. It may belong in a different repository.
from collections import namedtuple
from multiprocessing import Process, Queue
import resource
import sys
import time
from typing import Any, Callable, List
from termcolor import colored
import traceback


_WorkflowResult = namedtuple("_WorkflowResult", ["name", "time", "max_memory", "exception"])


def get_jobs(objs: Any):
    """
    Get a list of functions from a class (or module or object) suitable for calling execute on.

    Example
    -------
    >>> class Jobs:
    ...     def job1(self:
    ...         print("Job 1")
    ...     def job2(self):
    ...         print("Job 2")
    >>> execute(get_jobs(Jobs))
    """
    return [
        getattr(objs, attr)
        for attr in dir(objs)
        if callable(getattr(objs, attr)) and not attr.startswith("__")
    ]


def _runjob(job: Callable, queue: Queue):
    workflow_runner = _WorkflowRunner(job.__name__)
    with workflow_runner:
        job()
    queue.put(workflow_runner.result)


def execute(jobs: List[Callable], verbose=False):
    """
    Run a list of jobs in separate processes (one at a time). Print a summary of results.

    Example
    -------
    >>> class Jobs:
    ...     def job1(self):
    ...         print("Job 1")
    ...     def job2(self):
    ...         print("Job 2")
    >>> execute(get_jobs(Jobs))
    """
    assert len(jobs) > 0, "Nothing to run"
    if verbose:
        print(f"Running {len(jobs)} jobs")

    # In Python 3.11+ we could use concurrent.future as follows.
    #   with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
    # Since ProcessPoolExecutor doesn't have max_tasks_per_child yet, we use multiprocessing.
    results = []
    for job in jobs:
        queue = Queue()
        process = Process(target=_runjob, args=(job, queue))
        process.start()
        process.join()
        try:
            result = queue.get()
        except TypeError:
            # sys.exc_info() returns a tuple of three values that give
            # information about the exception that is currently being handled.
            result = _WorkflowResult(job.__name__, 0, 0, sys.exc_info())
        results.append(result)
    _summarize(results)


class _WorkflowRunner:
    def __init__(self, name: str):
        """
        Parameters
        ----------
        name: str
            Name of the workflow
        """
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        # kB on Linux, bytes on macOS
        max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.result = _WorkflowResult(self.name, time.time() - self.start, max_mem, args)
        return True


def _stat(result: _WorkflowResult):
    if result.max_memory < 1_000_000:
        memstr = ""
    elif sys.platform == "linux":
        mem_GB = result.max_memory / 1_000_000.0  # kB to GB
        memstr = f"({mem_GB:.1f}GB)"
    elif sys.platform == "darwin":
        mem_GB = result.max_memory / 1_000_000_000.0  # B to GB
        memstr = f"({mem_GB:.1f}GB)"
    else:
        # Uncertain what units ru_maxrss has
        memstr = result.max_memory
    return f"({result.time:.1f}s) {memstr}"


def _summarize(results: List[_WorkflowResult]):
    """
    Print a summary of successful/failed workflows, with stacktraces
    """
    successes = [result for result in results if not result.exception[2]]
    failures = [result for result in results if result.exception[2]]
    success_count = ""
    failure_count = ""

    for result in failures:
        print(colored(f"====== {result.name} ======", "red"))
        traceback.print_exception(*result.exception)

    if len(successes) > 0:
        print(colored("====== Successes ======", "green"))
        success_count = f"{len(successes)} succeeded"
    for result in successes:
        print(colored(result.name, "green"), colored(_stat(result), "yellow"))

    if len(failures) > 0:
        print(colored("====== Failures ======", "red"))
        failure_count = f"{len(failures)} failed"
    for result in failures:
        exc_type, exc_value, _ = result.exception
        msg = f"{result.name} {_stat(result)}. {exc_type.__name__} {exc_value}. See stack trace above."
        print(colored(msg, "red"))

    if len(failures) > 0:
        final_color = "red"
    else:
        final_color = "green"
    print(
        colored("======", final_color),
        colored(success_count, "green"),
        colored(failure_count, "red"),
        colored("======", final_color),
    )
