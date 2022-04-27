"""
Trying out some parallel processing
"""


from concurrent.futures import ProcessPoolExecutor
from optimization.general.timer import *


def foo(a, b):
    """
    Sum two elements.
    """
    c = a + b
    return a, b, c


if __name__ == '__main__':

    timer = Timer()

    solutions = []
    n = 100000
    alist = list(range(0, n, 1))
    blist = list(range(0, n, 1))

    with ProcessPoolExecutor() as executor:

        futures = []
        for idx, _ in enumerate(alist):
            solution = executor.submit(foo, alist[idx], blist[idx])
            futures.append(solution)

        for future in futures:
            solution = future.result()
            solutions.append(solution)

    print('parallel')
    timer.stop()

    # ##############

    timer = Timer()

    solutions = []

    for idx, _ in enumerate(alist):
        solution = foo(alist[idx], blist[idx])
        solutions.append(solution)

    print('sequential')
    timer.stop()
