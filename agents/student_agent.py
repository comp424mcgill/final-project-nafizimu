# Student agent: Add your own agent here
import random
from typing import Any, Dict, List, Tuple, Union
from xmlrpc.client import MAXINT
from agents.agent import Agent
from store import register_agent
import queue
import sys
import math
import numpy as np
import heapq


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        self.directions = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        return self.alpha_beta_pruning(
            chess_board, my_pos, adv_pos, 2, 2, max_step, True, 10, -MAXINT, MAXINT
        )

    def depth_limited_search(
        self,
        chess_board,
        depth: int,
        start: Tuple[int, int],
        *adv_pos: Tuple[Tuple[int, int], ...],
    ):
        MOVES: List[Tuple[int, Tuple[int, int]]] = list(
            enumerate(self.directions)
        )
        random.shuffle(MOVES)

        class StackFrame:
            def __init__(self, start: Tuple[int, int]) -> None:
                self.start = start
                self.it = iter(MOVES)

        stack = [StackFrame(start)]
        path = [start]
        visited = {start}

        if len(path) > depth:
            raise StopIteration()
        else:
            yield path

        while stack:
            top = stack[-1]
            start = top.start
            it = top.it

            try:
                if len(path) > depth:
                    raise StopIteration()

                # find a neighbor
                i, move = next(it)
                pos = (start[0] + move[0], start[1] + move[1])
                if (
                    not chess_board[start[0]][start[1]][i]
                    and pos not in visited
                    and pos not in adv_pos
                    and pos[0] >= 0
                    and pos[0] < len(chess_board)
                    and pos[1] >= 0
                    and pos[1] < len(chess_board)
                ):
                    path.append(pos)
                    visited.add(pos)
                    stack.append(StackFrame(pos))
                    yield path
            except StopIteration:
                # current path exhausted
                path.pop()
                visited.remove(top.start)
                stack.pop()

    def monte_carlo_method(
        self, chess_board, my_pos: Tuple[int, int], adv_pos: Tuple[int, int], max_step
    ):
        class StackFrame:
            def __init__(
                self, my_pos: Tuple[int, int], adv_pos: Tuple[int, int], dir: int = None
            ) -> None:
                self.my_pos = my_pos
                self.adv_pos = adv_pos
                self.dir = dir

        stack = [StackFrame(my_pos, adv_pos)]
        while stack:
            my_pos = stack[-1].my_pos
            adv_pos = stack[-1].adv_pos

            if (
                score := self.game_score(chess_board, my_pos, adv_pos)
            ) is not None:
                # undo all walls created (the first item is the initial state)
                for item in stack[1:]:
                    # adv_pos is lucky_pos as can be seen at the end of the outer loop
                    self.set_wall(chess_board, item.adv_pos, item.dir, False)

                    # swap min and max
                    score = (score[1], score[0])
                return score

            lucky_pos = random.choice(
                [
                    path[-1]
                    for path in self.depth_limited_search(
                        chess_board, max_step, my_pos, adv_pos
                    )
                ]
            )

            lucky_dir = random.choice(
                [
                    i
                    for i, wall in enumerate(chess_board[lucky_pos[0]][lucky_pos[1]])
                    if not wall
                ]
            )

            self.set_wall(chess_board, lucky_pos, lucky_dir, True)
            stack.append(StackFrame(adv_pos, lucky_pos, lucky_dir))

        raise Exception("Supposed to return score in the while loop")

    def game_score(self, chess_board, my_pos, adv_pos, isAdv=False):
        def dist(a, b):
            return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        MOVES = list(enumerate(self.directions))
        total_tiles = len(chess_board) * len(chess_board[0])

        tocheck: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(tocheck, (dist(my_pos, adv_pos), my_pos))

        checked = set()
        checked.add(my_pos)

        while not tocheck:

            _, cur_pos = heapq.heappop(tocheck)

            for m in MOVES:
                new_row = cur_pos[0] + m[1][0]
                new_col = cur_pos[1] + m[1][1]

                if (
                    new_row >= len(chess_board)
                    or new_col >= len(chess_board[new_row])
                    or new_row < 0
                    or new_col < 0
                ):
                    continue

                dir = m[0]

                if (new_row, new_col) == adv_pos and not chess_board[cur_pos[0]][
                    cur_pos[1]
                ][dir]:
                    return None

                if (
                    not chess_board[cur_pos[0]][cur_pos[1]][dir]
                    and (new_row, new_col) not in checked
                ):
                    new_pos = new_row, new_col
                    heapq.heappush(tocheck, (dist(new_pos, adv_pos), new_pos))
                    checked.add((new_row, new_col))

        total_visited = len(checked)

        try:
            if total_visited <= 1:
                raise Exception()
        except:
            pass

        if len(checked) == total_tiles:
            return None
        elif not isAdv:
            return (
                total_visited,
                self.game_score(chess_board, adv_pos, my_pos, True)[0],
            )
        elif isAdv:
            return (total_visited, -1)

    def alpha_beta_pruning(
        self,
        chess_board,
        my_pos,
        adv_pos,
        ab_depth,  # how deep ab_pruning will go
        max_ab_depth,
        max_step,  # max step allowed in this game
        isMaxPlayer,
        mcm_numbers,  # how many random simulations to do
        alpha,  # max, start with -inf
        beta,  # min, start with inf
    ):
        pathes = self.depth_limited_search(  # get all the possible final points
            chess_board, max_step, my_pos, adv_pos
        )
        end_points = set()  # set of tuple ((int row, int col), int direction)

        for path in pathes:
            for i in range(4):  # loop to add walls
                if not chess_board[path[-1][0]][path[-1][1]][i]:
                    end_points.add((path[-1], i))

        if len(end_points) == 0:
            return (alpha, beta)

        if ab_depth == 1:
            a = alpha
            b = beta

            for item in end_points:
                # Compute win rate after following 'item'
                self.set_wall(chess_board, item[0], item[1], True)
                win_rate = self.get_win_rate(
                    chess_board, mcm_numbers, my_pos, adv_pos, max_step
                )
                self.set_wall(chess_board, item[0], item[1], False)

                # update alpha and beta depend on level
                if isMaxPlayer:
                    a = win_rate if win_rate > a else a
                    if a > beta:
                        return (alpha, beta)
                else:
                    b = win_rate if win_rate < b else b
                    if alpha > b:
                        return (alpha, beta)

            return (a, b)

        else:
            a = alpha
            b = beta
            best_point = next(iter(end_points))

            for item in end_points:
                # for each possible end point, do ab pruning on those to see which one has a better win rate
                self.set_wall(chess_board, item[0], item[1], True)
                result = self.alpha_beta_pruning(
                    chess_board,
                    adv_pos,
                    item[0],  # my_pos
                    ab_depth - 1,
                    max_ab_depth,
                    max_step,
                    not isMaxPlayer,  # flip the isMaxPlayer because it's the opponant turn.
                    mcm_numbers,
                    a,
                    b,
                )
                self.set_wall(chess_board, item[0], item[1], False)

                if isMaxPlayer:
                    if result[1] > a:
                        a = result[1]
                        best_point = item
                    if a > beta:
                        return (alpha, beta)
                else:
                    if result[0] < b:
                        b = result[0]
                        best_point = item
                    if alpha > b:
                        return (alpha, beta)

            if ab_depth == max_ab_depth:
                return best_point

            return (a, b)

    def get_win_rate(self, chess_board, mcm_numbers, my_pos, adv_pos, max_step):
        win_cnt = 0
        for _ in range(mcm_numbers):
            result = self.monte_carlo_method(
                chess_board, my_pos, adv_pos, max_step
            )
            if result[0] > result[1]:
                win_cnt = win_cnt + 1

        return win_cnt / mcm_numbers

    def set_wall(self, chess_board, pos, dir: int, wall: bool):
        chess_board[pos[0], pos[1], dir] = wall

        moves = self.directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        anti_pos = np.array(pos) + np.array(moves[dir])
        if all(0 <= anti_pos) and all(anti_pos < len(chess_board)):
            anti_dir = opposites[dir]
            chess_board[anti_pos[0], anti_pos[1], anti_dir] = wall
    
    def disjoint_sets(self, chess_board) -> Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
        sets: List[List[Tuple[int, int]]] = [
            [None] * len(chess_board[i]) for i in range(len(chess_board))
        ]
        counts: Dict[Tuple[int, int], int] = dict()
        for i in range(len(chess_board)):
            for j in range(len(chess_board[i])):
                if sets[i][j] is None:
                    # enter new territory
                    sets[i][j] = (i, j)
                    counts[sets[i][j]] = 0

                counts[sets[i][j]] += 1

                # right accessible?
                if not chess_board[i][j][1]:
                    sets[i][j + 1] = sets[i][j]
                # down accessible?
                if not chess_board[i][j][2]:
                    sets[i + 1][j] = sets[i][j]

        return sets, counts
