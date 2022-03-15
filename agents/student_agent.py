# Student agent: Add your own agent here
import random
from typing import List, Tuple, Union
from agents.agent import Agent
from store import register_agent
import sys


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
        # dummy return
        return my_pos, self.dir_map["u"]

    @staticmethod
    def dls(a: Tuple[int, int], adv_pos: Tuple[int, int], chess_board, depth: int):
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

        if len(path) > depth:
            raise StopIteration()
        else:
            yield path

        while stack:
            top = stack[-1]
            a = top.a
            it = top.it

            try:
                if len(path) > depth:
                    raise StopIteration()

                # find a neighbor
                i, move = next(it)
                pos = (a[0] + move[0], a[1] + move[1])
                if (
                    not chess_board[a[0]][a[1]][i]
                    and pos not in visited
                    and pos != adv_pos
                ):
                    path.append(pos)
                    visited.add(pos)
                    stack.append(StackFrame(pos))
                    yield path
            except StopIteration:
                # current path exhausted
                path.pop()
                visited.remove(top.a)
                stack.pop()

    @staticmethod
    def game_score(
        chess_board, my_pos: Tuple[int, int], adv_pos: Tuple[int, int]
    ) -> Union[Tuple[int, int], None]:
        return None

    @staticmethod
    def mcts(chess_board, my_pos: Tuple[int, int], adv_pos: Tuple[int, int], max_step):
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
                score := StudentAgent.game_score(chess_board, my_pos, adv_pos)
            ) is not None:
                # undo all walls created (the first item is the initial state)
                for item in stack[1:]:
                    # adv_pos is lucky_pos as can be seen at the end of the outer loop
                    chess_board[item.adv_pos[0]][item.adv_pos[1]][item.dir] = False
                return score

            # max_step + 1 to include root
            lucky_pos = random.choice(
                [
                    path[-1]
                    for path in StudentAgent.dls(
                        my_pos, adv_pos, chess_board, max_step + 1
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

            chess_board[lucky_pos[0]][lucky_pos[1]][lucky_dir] = True
            stack.append(StackFrame(adv_pos, lucky_pos, lucky_dir))

        raise Exception("Supposed to return score in the while loop")
