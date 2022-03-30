# Student agent: Add your own agent here
from collections import deque
from dataclasses import dataclass, field
from functools import total_ordering
import random
import sys
from typing import Dict, List, Tuple

from matplotlib.pyplot import connect
from agents.agent import Agent
from store import register_agent
import numpy as np
import heapq

MAX_ROUND = 10 * 10 * 4

@total_ordering
@dataclass
class MCTSNode:
    my_pos: Tuple[int, int] = field(compare=False, default=(-1, -1))
    adv_pos: Tuple[int, int] = field(compare=False, default=(-1, -1))
    my_dir: int = field(compare=False, default=-1)
    win: int = field(default=0)
    draw: int = field(default=0)
    round: int = field(default=0)
    parent: "MCTSNode" = field(compare=False, default=None)
    children: List["MCTSNode"] = field(compare=False, default_factory=list)

    def __lt__(self, other: "MCTSNode"):
        return self.win/self.round > other.win/self.round
    
    def default_policy(self, chess_board, max_step):
        for _ in range(10):
            adv_pos, adv_dir, adv_score = StudentAgent.monte_carlo_method(chess_board, self.adv_pos, self.my_pos, max_step)
            
            self.round += 1
            self.win += adv_score[1] > adv_score[0]
            self.draw += adv_score[0] == adv_score[1]
            
            key = (adv_pos, adv_dir)
            if key not in self.children:
                self.children[key] = MCTSNode(self, adv_pos, adv_dir, self.my_pos)
            
            scores[key][0] += 1
            scores[key][1] += adv_score[0] > adv_score[1]
            scores[key][2] += adv_score[0] == adv_score[1]


    def tree_policy(self, chess_board, max_step):
        pass

def mcts(chess_board, my_pos, adv_pos, max_step):
    pass

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))

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
        return StudentAgent.alpha_beta_pruning(
            chess_board,
            my_pos,
            adv_pos,
            2,
            2,
            max_step,
            True,
            20,
            -sys.maxsize,
            sys.maxsize,
        )

    @staticmethod
    def bfs(
        chess_board,
        my_pos: Tuple[int, int],
        max_step: int = 100,
        adv_pos: Tuple[int, int] = None,
    ):
        MOVES: List[Tuple[int, Tuple[int, int]]] = list(enumerate(StudentAgent.directions))
        random.shuffle(MOVES)

        q = deque([(0, my_pos)])
        visited = {my_pos}

        while q:
            step, cur_pos = q.popleft()
            step: int = step
            cur_pos: Tuple[int, int] = cur_pos
            yield step, cur_pos

            if step >= max_step:
                return

            # find neighbors
            for i, move in MOVES:
                new_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if (
                    not chess_board[cur_pos[0]][cur_pos[1]][i]
                    and new_pos not in visited
                    and new_pos != adv_pos
                    and new_pos[0] >= 0
                    and new_pos[0] < len(chess_board)
                    and new_pos[1] >= 0
                    and new_pos[1] < len(chess_board)
                ):
                    # assert not all(chess_board[new_pos[0]][new_pos[1]])
                    q.append((step + 1, new_pos))
                    visited.add(new_pos)

    @staticmethod
    def monte_carlo_method(
        chess_board,
        my_pos: Tuple[int, int],
        adv_pos: Tuple[int, int],
        max_step,
        max_round=MAX_ROUND,
    ):
        class StackFrame:
            def __init__(
                self, my_pos: Tuple[int, int], adv_pos: Tuple[int, int], dir: int = None
            ) -> None:
                self.my_pos = my_pos
                self.adv_pos = adv_pos
                self.dir = dir

        stack: List[StackFrame] = []
        walls_connected = False

        while True:
            my_pos = stack[-1].my_pos if stack else my_pos
            adv_pos = stack[-1].adv_pos if stack else adv_pos

            if (walls_connected) or all(chess_board[my_pos]) or len(stack) >= max_round:
                score = StudentAgent.game_score(chess_board, my_pos, adv_pos)
                if not score and len(stack) >= max_round:
                    score = (1, 1)

                if score is not None:
                    # undo all walls created (the first item is the initial state)
                    for item in stack:
                        # adv_pos is lucky_pos as can be seen at the end of the outer loop
                        StudentAgent.set_wall(chess_board, item.adv_pos, item.dir, False)

                    if not stack:
                        return my_pos, 0, score

                    # swap min and max
                    if len(stack) % 2 == 0:
                        score = (score[1], score[0])
                    
                    return stack[0].my_pos, stack[0].dir, score

            lucky_pos = random.choice(
                [p for _, p in StudentAgent.bfs(chess_board, my_pos, max_step, adv_pos)]
            )

            lucky_dir = random.choice(
                [
                    i
                    for i, wall in enumerate(chess_board[lucky_pos[0]][lucky_pos[1]])
                    if not wall
                ]
            )

            walls_connected = StudentAgent.set_wall(chess_board, lucky_pos, lucky_dir, True)
            stack.append(StackFrame(adv_pos, lucky_pos, lucky_dir))

    @staticmethod
    def greedy_search(
        chess_board, a: Tuple[int, int], b: Tuple[int, int], end_at_b=False
    ):
        def dist(a, b):
            return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        MOVES = list(enumerate(StudentAgent.directions))

        tocheck: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(tocheck, (dist(a, b), a))

        checked = set()
        checked.add(a)

        while tocheck:
            _, cur_pos = heapq.heappop(tocheck)

            yield cur_pos

            for m in MOVES:
                new_row = cur_pos[0] + m[1][0]
                new_col = cur_pos[1] + m[1][1]
                new_pos = new_row, new_col

                if (
                    new_row >= len(chess_board)
                    or new_col >= len(chess_board[new_row])
                    or new_row < 0
                    or new_col < 0
                ):
                    continue

                dir = m[0]

                if (
                    not chess_board[cur_pos[0]][cur_pos[1]][dir]
                    and new_pos not in checked
                ):
                    if end_at_b and new_pos == b:
                        yield b
                        return

                    heapq.heappush(tocheck, (dist(new_pos, b), new_pos))
                    checked.add(new_pos)

    @staticmethod
    def game_score(chess_board, my_pos, adv_pos, isAdv=False):
        total_tiles = len(chess_board) * len(chess_board[0])
        total_visited = 0

        for pos in StudentAgent.greedy_search(chess_board, my_pos, adv_pos, end_at_b=True):
            if pos == adv_pos:
                return None

            total_visited += 1

        # try:
        #     if total_visited <= 1:
        #         raise Exception()
        # except:
        #     pass

        # i may have forgotten, but this if statement makes me a bit sus...
        assert total_visited != total_tiles
        if not isAdv:
            return (
                total_visited,
                StudentAgent.game_score(chess_board, adv_pos, my_pos, True)[0],
            )
        elif isAdv:
            return (total_visited, -1)

    @staticmethod
    def alpha_beta_pruning(
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
        end_points = [
            (point, i)
            for (_, point) in StudentAgent.bfs(chess_board, my_pos, max_step, adv_pos)
            for i in range(4)
            if not chess_board[point[0]][point[1]][i]
        ]
        random.shuffle(end_points)
        end_points = end_points[:25]

        if len(end_points) == 0:
            return (alpha, beta)

        if ab_depth == 1:
            a = alpha
            b = beta

            for item in end_points:
                # Compute win rate after following 'item'
                StudentAgent.set_wall(chess_board, item[0], item[1], True)
                win_rate = StudentAgent.get_win_rate(
                    chess_board, mcm_numbers, my_pos, adv_pos, max_step
                )
                StudentAgent.set_wall(chess_board, item[0], item[1], False)

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
                StudentAgent.set_wall(chess_board, item[0], item[1], True)
                result = StudentAgent.alpha_beta_pruning(
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
                StudentAgent.set_wall(chess_board, item[0], item[1], False)

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

    @staticmethod
    def get_win_rate(chess_board, mcm_numbers, my_pos, adv_pos, max_step):
        win_cnt = 0
        for _ in range(mcm_numbers):
            _, _, result = StudentAgent.monte_carlo_method(chess_board, my_pos, adv_pos, max_step)
            if result[0] > result[1]:
                win_cnt = (
                    win_cnt + (result[0] > result[1]) + (result[0] == result[1]) * 0.5
                )

        return win_cnt / mcm_numbers

    @staticmethod
    def set_wall(chess_board, pos, dir: int, wall: bool):
        # assert chess_board[pos[0], pos[1], dir] != wall
        chess_board[pos[0], pos[1], dir] = wall

        moves = StudentAgent.directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        anti_pos = np.array(pos) + np.array(moves[dir])
        anti_dir = opposites[dir]

        # assert chess_board[anti_pos[0], anti_pos[1], anti_dir] != wall
        chess_board[anti_pos[0], anti_pos[1], anti_dir] = wall

        neighbor_l_pos = np.array(pos) + np.array(moves[(dir - 1) % 4])
        parallel_left = (
            chess_board[neighbor_l_pos[0], neighbor_l_pos[1], dir]
            if all(len(chess_board) > neighbor_l_pos) and all(neighbor_l_pos >= 0)
            else False
        )
        neighbor_r_pos = np.array(pos) + np.array(moves[(dir + 1) % 4])
        parallel_right = (
            chess_board[neighbor_r_pos[0], neighbor_r_pos[1], dir]
            if all(len(chess_board) > neighbor_r_pos) and all(neighbor_r_pos >= 0)
            else False
        )

        return (
            parallel_left
            or chess_board[pos[0], pos[1], (dir - 1) % 4] == True
            or chess_board[anti_pos[0], anti_pos[1], (dir - 1) % 4] == True
        ) and (
            parallel_right
            or chess_board[pos[0], pos[1], (dir + 1) % 4] == True
            or chess_board[anti_pos[0], anti_pos[1], (dir + 1) % 4] == True
        )

    @staticmethod
    def disjoint_sets(
        chess_board
    ) -> Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
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
