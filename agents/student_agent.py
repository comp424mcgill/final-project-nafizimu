# Student agent: Add your own agent here
from collections import deque
import random
import sys
from typing import Dict, List, Tuple

from matplotlib.pyplot import connect
from agents.agent import Agent
from store import register_agent
import numpy as np
import heapq
import time

MAX_ROUND = 10 * 10 * 4
MONTE_CARLO_CNT = 10
UCT_CONST = np.sqrt(2)
TWO_SEC = (2 * 8 * 10**9) // 10
THIRTY_SEC = (30 * 9 * 10**9) // 10
MIN_BRANCH = 5


class MCTSNode:
    def __init__(
        self, my_pos, my_dir, adv_pos, chess_board, max_step, parent=None, is_adv=False
    ) -> None:
        self.my_pos: Tuple[int, int] = my_pos
        self.my_dir: int = my_dir
        self.adv_pos: Tuple[int, int] = adv_pos
        self.win: int = 0
        self.round: int = 0
        self.parent: "MCTSNode" = parent
        self.children: List["MCTSNode"] = []
        self.is_adv = is_adv
        self.score: Tuple[int, int] = None

        d_round = 0
        d_win = 0

        # start of default policy

        if self.parent:
            StudentAgent.set_wall(chess_board, self.my_pos, self.my_dir, True)

        self.score = StudentAgent.game_score(chess_board, self.my_pos, self.adv_pos)
        if self.score:
            d_round += MONTE_CARLO_CNT
            d_win += (self.score[0] > self.score[1]) * MONTE_CARLO_CNT
        else:
            for _ in range(MONTE_CARLO_CNT):
                adv_score, my_score = self.monte_carlo_method(chess_board, max_step)
                d_round += 1
                d_win += my_score > adv_score

        if self.parent:
            StudentAgent.set_wall(chess_board, self.my_pos, self.my_dir, False)

        self.round += d_round
        d_win = d_win if not self.is_adv else d_round - d_win
        self.win += d_win
        self.back_propagation(d_round, d_win)

    def tree_policy(
        self, chess_board, max_step
    ):
        """
        Performs tree policy to update the win-rate of the most optimal path
        """
        path = [self]
        while path[-1].children:
            best_node = path[-1].best_child()
            path.append(best_node)
            StudentAgent.set_wall(chess_board, best_node.my_pos, best_node.my_dir, True)

        if not path[-1].parent:
            end_points = [
                (point, i)
                for (_, point) in StudentAgent.bfs(
                    chess_board, self.my_pos, max_step, self.adv_pos
                )  # all the children all self
                for i in range(4)
                if not chess_board[point[0]][point[1]][i]
            ]
            random.shuffle(end_points)

            for (point, i) in end_points[: max(len(end_points) // 2, MIN_BRANCH)]:
                new_child = MCTSNode(
                    point, i, self.adv_pos, chess_board, max_step, self, self.is_adv
                )
                self.children.append(new_child)
                if time.time_ns() >= StudentAgent.end_time:
                    return
        else:
            leaf = path[-1]

            end_points = [
                (point, i)
                for (_, point) in StudentAgent.bfs(
                    chess_board, leaf.adv_pos, max_step, leaf.my_pos
                )  # all the children all self
                for i in range(4)
                if not chess_board[point[0]][point[1]][i]
            ]
            random.shuffle(end_points)

            for (point, i) in end_points[: max(len(end_points) // 2, MIN_BRANCH)]:
                new_child = MCTSNode(
                    point, i, leaf.my_pos, chess_board, max_step, leaf, not leaf.is_adv
                )
                leaf.children.append(new_child)
                if time.time_ns() >= StudentAgent.end_time:
                    return

        # first item is root
        for node in path[1:]:
            StudentAgent.set_wall(chess_board, node.my_pos, node.my_dir, False)

    def back_propagation(self, d_round, d_win):
        """
        Performs back-propagation to update win-ratio of all predecessors
        """
        node = self
        while node.parent:
            node.parent.win += d_win
            node.parent.round += d_round
            node = node.parent

    def best_child(self, greedy=False):
        """
        Returns the best child. If greedy is True, return the child with highest win-rate.
        Otherwise, pick the next child based on UCT.
        """
        def uct_cal(child: "MCTSNode"):  # cur/next_data: (win_cnt, total_round)
            if greedy:
                return child.win / child.round
            else:
                return child.win / child.round + UCT_CONST * np.sqrt(
                    np.log(self.round) / child.round
                ) * (not child.score)

        return max(self.children, key=uct_cal)

    def monte_carlo_method(
        self,
        chess_board,
        max_step,
        max_round=MAX_ROUND,
    ):
        """
        performs monte carlo method
        assumes that chess_board[my_pos[0]][my_pos[1]][dir] == True
        """

        class StackFrame:
            def __init__(
                self, my_pos: Tuple[int, int], dir: int, adv_pos: Tuple[int, int]
            ) -> None:
                self.my_pos = my_pos
                self.adv_pos = adv_pos
                self.my_dir = dir

        stack: List[StackFrame] = []
        walls_connected = True

        while True:
            my_pos = stack[-1].adv_pos if stack else self.adv_pos
            adv_pos = stack[-1].my_pos if stack else self.my_pos

            if (
                walls_connected
                or all(chess_board[my_pos])
                or all(chess_board[adv_pos])
                or len(stack) >= max_round
            ):
                score = StudentAgent.game_score(chess_board, my_pos, adv_pos)

                if not score and len(stack) >= max_round:
                    score = (1, 1)

                if score is not None:
                    for item in stack:
                        StudentAgent.set_wall(
                            chess_board, item.my_pos, item.my_dir, False
                        )

                    if not stack:
                        return score

                    # swap min and max
                    if len(stack) % 2 == 1:
                        score = (score[1], score[0])

                    return score

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

            walls_connected = StudentAgent.set_wall(
                chess_board, lucky_pos, lucky_dir, True
            )
            stack.append(StackFrame(lucky_pos, lucky_dir, adv_pos))

    # Codes used to debug MCTS visually

    # def __str__(self) -> str:
    #     return f"{self.my_pos}:{self.my_dir}:{self.adv_pos}:{self.is_adv}:{self.win}/{self.round}"

    # def __repr__(self) -> str:
    #     return str(self)

    # def __str_for_node(self):
    #     return (
    #         f"{self.my_pos} {self.my_dir} vs. {self.adv_pos}\n{self.win}/{self.round}"
    #     )

    # def dfs(self):
    #     stack: List[MCTSNode] = [self]
    #     while stack:
    #         top = stack.pop()
    #         yield top
    #         stack.extend(top.children)

    # def tree_to_text(self, filename: str = "tree.txt"):
    #     from treelib import Tree

    #     tree = Tree()

    #     for node in self.dfs():
    #         tree.create_node(
    #             str(node), id(node), parent=id(node.parent) if node.parent else None
    #         )

    #     tree.save2file(filename)

    # def to_svg(self, filename: str = "tree"):
    #     import graphviz

    #     dot = graphviz.Digraph()
    #     dot.attr(overlap="false")
    #     dot.attr(ranksep="1")

    #     for node in self.dfs():
    #         dot.node(str(id(node)), node.__str_for_node())
    #         if node.parent:
    #             dot.edge(str(id(node.parent)), str(id(node)))

    #     dot.render(filename, cleanup=True, format="svg")


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    end_time = 0

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
        self.is_first_round = True

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
        # return StudentAgent.alpha_beta_pruning(
        #     chess_board,
        #     my_pos,
        #     adv_pos,
        #     2,
        #     2,
        #     max_step,
        #     True,
        #     20,
        #     -sys.maxsize,
        #     sys.maxsize,
        # )

        if self.is_first_round:
            self.is_first_round = False
            StudentAgent.end_time = time.time_ns() + THIRTY_SEC
            return StudentAgent.mcts(chess_board, my_pos, adv_pos, max_step)
        else:
            StudentAgent.end_time = time.time_ns() + TWO_SEC
            return StudentAgent.mcts(chess_board, my_pos, adv_pos, max_step)

    @staticmethod
    def bfs(
        chess_board,
        my_pos: Tuple[int, int],
        max_step: int = 100,
        adv_pos: Tuple[int, int] = None,
    ):
        """
        Performs Breath-first Search and returns a sequence of (# of step, position) from the my_pos up to max_step.
        If adv_pos given, then it will avoid adv_pos
        """
        MOVES: List[Tuple[int, Tuple[int, int]]] = list(
            enumerate(StudentAgent.directions)
        )
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
    def greedy_search(chess_board, a: Tuple[int, int], b: Tuple[int, int]):
        """
        Performs greedy search and returns a sequence of (# of steps, position) starting at 'a'.
        It will pick the next position closest to 'b'.
        There is no step limit, so it can explore the entire region.
        """
        def dist(a, b):
            return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        MOVES = list(enumerate(StudentAgent.directions))

        tocheck: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(tocheck, (dist(a, b), 0, a))

        checked = set()
        checked.add(a)

        while tocheck:
            _, step, cur_pos = heapq.heappop(tocheck)

            yield (step, cur_pos)

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
                    heapq.heappush(tocheck, (dist(new_pos, b), step + 1, new_pos))
                    checked.add(new_pos)

    @staticmethod
    def game_score(chess_board, my_pos, adv_pos, isAdv=False):
        """
        It returns the score of players at my_pos and adv_pos as a tuple if they are on separate regions.
        Otherwise, it returns None.
        """
        total_tiles = len(chess_board) * len(chess_board[0])
        total_visited = 0

        for (_, pos) in StudentAgent.greedy_search(chess_board, my_pos, adv_pos):
            if pos == adv_pos:
                return None

            total_visited += 1

        # try:
        #     if total_visited <= 1:
        #         raise Exception()
        # except:
        #     pass

        assert total_visited != total_tiles
        if not isAdv:
            return (
                total_visited,
                StudentAgent.game_score(chess_board, adv_pos, my_pos, True)[0],
            )
        elif isAdv:
            return (total_visited, -1)

    @staticmethod
    def set_wall(chess_board, pos, dir: int, wall: bool):
        """
        Sets the wall at pos towards dir.
        Sets the second wall at the opposite position and direction.
        It returns True if both ends of the wall is connected to another wall.
        """
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
    def mcts(chess_board, my_pos, adv_pos, max_step):
        """
        Performs mcts and returns the best next position and direction discovered.
        """

        # start_time = time.time_ns()
        root = MCTSNode(my_pos, -1, adv_pos, chess_board, max_step)

        while time.time_ns() < StudentAgent.end_time:
            # while True:
            # root.to_svg()
            root.tree_policy(chess_board, max_step)

        # print("time: " + str((time.time_ns() - start_time) / (10**9)))

        best_point = root.best_child(True)
        return (best_point.my_pos, best_point.my_dir)
