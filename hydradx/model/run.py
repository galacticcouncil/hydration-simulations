import time

from .amm.global_state import GlobalState
from copy import deepcopy


def run(initial_state: GlobalState, time_steps: int, silent: bool = False) -> list:
    """
    Definition:
    Run simulation
    """

    start_time = time.time()
    events = []
    new_global_state = initial_state.copy()

    if not silent:
        print('Starting simulation...')

    for i in range(time_steps):

        # market evolutions
        new_global_state.update()

        events.append(new_global_state.archive())

    if not silent:
        print(f'Execution time: {round(time.time() - start_time, 3)} seconds.')
    return events
