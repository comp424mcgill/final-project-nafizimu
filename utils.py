from contextlib import contextmanager
from enum import Enum, auto
import logging
import random
from typing import List, Tuple


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)

def dfs(a: Tuple[int, int], chess_board):
    MOVES: List[Tuple[int, Tuple[int, int]]] = list(
        enumerate((-1, 0), (0, 1), (1, 0), (0, -1))
    )
    random.shuffle(MOVES)

    class StackFrame:
        def __init__(self, a: Tuple[int, int]) -> None:
            self.a = a
            self.it = iter(MOVES)

    stack = [StackFrame(a)]
    path = [a]
    visited = {a}

    # stop if depth too high
    stop = bool((yield path))
    if stop:
        raise StopIteration()

    while stack:
        top = stack[-1]
        a = top.a
        it = top.it

        try:
            # find a neighbor
            i, move = next(it)
            pos = (a[0] + move[0], a[1] + move[1])
            if not chess_board[a[0]][a[1]][i] and pos not in visited:
                path.append(pos)
                visited.add(pos)
                stack.append(StackFrame(pos))

                # stop if depth too high
                stop = bool((yield path))
                if stop:
                    raise StopIteration()
        except StopIteration:
            # current path exhausted
            path.pop()
            visited.remove(top.a)
            stack.pop()

def dls(a: Tuple[int, int], chess_board, depth: int):
    try:
        paths = dfs(a, chess_board)
        path = paths.send(None)
        while True:
            if len(path) > depth:
                path = paths.send(True)
            else:
                yield path
                path = paths.send(False)
    except StopIteration:
        pass
