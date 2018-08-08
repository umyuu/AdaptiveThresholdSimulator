# -*- coding: utf-8 -*-
"""
    StopWatch
    経過時間測定用
"""
from src.reporter import get_current_reporter
LOGGER = get_current_reporter()
from time import perf_counter
from traceback import extract_stack
from itertools import count

def stop_watch():

    start_time = perf_counter()
    endt_time = start_time
    c = count()

    def step():
        nonlocal endt_time
        func_name = extract_stack(None, 2)[0][2]
        n = next(c)
        elapsed = endt_time - start_time
        end_time = perf_counter()
        endt_time = end_time
        MSG = [func_name, n, end_time, elapsed]
        LOGGER.debug(MSG)
        return MSG, end_time - start_time

    return step


def main():
    """
        Entry Point
    """
    ct = stop_watch()
    ct()


if __name__ == "__main__":
    main()
