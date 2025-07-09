# connect4.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
#

"""
In this assignment, the task is to implement the adversarial search algorithms with 
depth limit for a Connect-4 game.

To complete the assignment, you must finish these functions:
    minimax (line 196), alphabeta (line 237), and expectimax (line 280).

In the Connect-4 game, two players place disks in a 6-by-7 board one by one.
The disks will fall straight down and occupy the lowest available space of
the chosen column. The player wins if four of his or her disks are connected
in a line horizontally, vertically or diagonally.
See https://en.wikipedia.org/wiki/Connect_Four for more details about the game.

A Board() class is provided to simulate the game board.
It has the following properties:
    b.rows          # number of rows of the game board
    b.cols          # number of columns of the game board
    b.PLAYER1       # an integer flag to represent the player 1
    b.PLAYER2       # an integer flag to represent the player 2
    b.EMPTY_SLOT    # an integer flag to represent an empty slot in the board;

and the following methods:
    b.terminal()            # check if the game is terminal
                            # terminal means draw or someone wins

    b.has_draw()            # check if the game is a draw

    w = b.who_wins()        # return the winner of the game or None if there
                            # is no winner yet 
                            # w should be in [b.PLAYER1,b.PLAYER2, None]

    b.occupied(row, col)    # check if the slot at the specific location is
                            # occupied

    x = b.get(row, col)     # get the player occupying the given slot
                            # x should be in [b.PLAYER1, b.PLAYER2, b.EMPTY_SLOT]

    row = b.row(r)          # get the specific row of the game described using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT

    col = b.column(r)       # get a specific column of the game board

    b.placeable(col)        # check if a disk can be placed at the specific
                            # column

    b.place(player, col)    # place a disk at the specific column for player
                            # raise ValueError if the specific column does not have available space
    
    new_board = b.clone()   # return a new board instance having the same
                            # disk placement with b

    str = b.dump()          # a string to describe the game board using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT
Hints:
    1. Depth-limited Search
        We use depth-limited search in the current code. That is, we
    stop the search forcefully, and perform evaluation directly not only when a
    terminal state is reached but also when the search reaches the specified
    depth. Here each depth denotes a single move by one player (ply). 
    2. Game State
        Three elements decide the game state. The current board state, the
    player that needs to take an action (place a disk), and the current search
    depth (remaining depth).
    3. Evaluation Target
        The minimax algorithm assumes that the adversary will always attempt 
    to minimize the score of the max player, for whom the algorithm is initially
    called. The adversary never considers its own score at all during this
    process. Therefore, when evaluating nodes by calling the evaluate() function, 
    the target should always be the max player.
    4. Search Result
        The pesudo code provided in the slides only returns the best utility value.
    However, in practice, we need to select the action that is associated with this
    value. Here, such action is specified as the column in which a disk should be
    placed for the max player. Therefore, for each search algorithm, you should
    consider all valid actions for the max player, and return the one that leads 
    to the best value. 

"""

# use math library if needed
import math

def get_child_boards(player, board):
    """
    Generate a list of successor boards obtained by placing a disk 
    at the given board for a given player
   
    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that will place a disk on the board
    board: the current board instance

    Returns
    -------
    a list of (col, new_board) tuples,
    where col is the column in which a new disk is placed (left column has a 0 index), 
    and new_board is the resulting board instance
    """
    res = []
    for c in range(board.cols):
        if board.placeable(c):
            tmp_board = board.clone()
            tmp_board.place(player, c)
            res.append((c, tmp_board))
    return res


def evaluate(player, board):
    """
    This is a function to evaluate the advantage of the specific player at the
    given game board. In your implementation, when calling evaluate(), player 
    should always be the MAX player.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the specific player
    board: the board instance

    Returns
    -------
    score: float
        a scalar to evaluate the advantage of the specific player at the given
        game board
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    # Initialize the value of scores
    # [s0, s1, s2, s3, --s4--]
    # s0 for the case where all slots are empty in a 4-slot segment
    # s1 for the case where the player occupies one slot in a 4-slot line, the rest are empty
    # s2 for two slots occupied
    # s3 for three
    # s4 for four
    score = [0]*5
    adv_score = [0]*5

    # Initialize the weights
    # [w0, w1, w2, w3, --w4--]
    # w0 for s0, w1 for s1, w2 for s2, w3 for s3
    # w4 for s4
    weights = [0, 1, 4, 16, 1000]

    # Obtain all 4-slot segments on the board
    seg = []
    invalid_slot = -1
    left_revolved = [
        [invalid_slot]*r + board.row(r) + \
        [invalid_slot]*(board.rows-1-r) for r in range(board.rows)
    ]
    right_revolved = [
        [invalid_slot]*(board.rows-1-r) + board.row(r) + \
        [invalid_slot]*r for r in range(board.rows)
    ]
    for r in range(board.rows):
        # row
        row = board.row(r) 
        for c in range(board.cols-3):
            seg.append(row[c:c+4])
    for c in range(board.cols):
        # col
        col = board.col(c) 
        for r in range(board.rows-3):
            seg.append(col[r:r+4])
    for c in zip(*left_revolved):
        # slash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    for c in zip(*right_revolved): 
        # backslash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    # compute score
    for s in seg:
        if invalid_slot in s:
            continue
        if adversary not in s:
            score[s.count(player)] += 1
        if player not in s:
            adv_score[s.count(adversary)] += 1
    reward = sum([s*w for s, w in zip(score, weights)])
    penalty = sum([s*w for s, w in zip(adv_score, weights)])
    return reward - penalty


def minimax(player, board, depth_limit):
    """
    Minimax algorithm with limited search depth.

    Parameters
    ----------
    player : Board.PLAYER1 or Board.PLAYER2
        The player (1 or 2) who is to move at the root of this call.
    board : Board
        The current game board instance.
    depth_limit : int
        Maximum remaining plies to search before calling evaluate().

    Returns
    -------
    placement : int or None
        The column (0-based) in which `player` should drop a disk.
        Returns None if there is no legal move (i.e., board.terminal() is True).
    """
    max_player = player
    placement = None

    def value(current_player, current_board, depth):
        # Base case: terminal state or depth = 0 → return evaluation
        if current_board.terminal() or depth == 0:
            return evaluate(player, current_board)

        if current_player == max_player:
            return max_value(current_player, current_board, depth)
        else:
            return min_value(current_player, current_board, depth)

    def max_value(current_player, current_board, depth):
        v = -math.inf
        # Iterate over all columns, check if placeable, then simulate
        for col in range(current_board.cols):
            if not current_board.placeable(col):
                continue
            # Clone & place
            child = current_board.clone()
            child.place(current_player, col)

            # Next to move is the other player:
            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1)
            if child_val > v:
                v = child_val
        return v

    def min_value(current_player, current_board, depth):
        v = math.inf
        # Opponent is “MIN” node (still trying to minimize max_player’s eventual score)
        for col in range(current_board.cols):
            if not current_board.placeable(col):
                continue
            child = current_board.clone()
            child.place(current_player, col)

            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1)
            if child_val < v:
                v = child_val
        return v

    # If the board is already terminal, no move is possible
    if board.terminal():
        return None

    # Root is a MAX node (since `player` is the one calling minimax)
    best_score = -math.inf
    for col in range(board.cols):
        if not board.placeable(col):
            continue
        child = board.clone()
        child.place(player, col)

        next_p = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
        child_score = value(next_p, child, depth_limit - 1)

        if child_score > best_score:
            best_score = child_score
            placement = col

    return placement



def alphabeta(player, board, depth_limit):
    """
    Minimax with alpha-beta pruning, depth-limited.

    Parameters
    ----------
    player : Board.PLAYER1 or Board.PLAYER2
        The “MAX” player to move at the root.
    board : Board
        The current game board.
    depth_limit : int
        How many plies to search before evaluating.

    Returns
    -------
    placement : int or None
        The chosen column (0-based), or None if board is already terminal.
    """
    max_player = player
    placement = None

    def value(current_player, current_board, depth, alpha, beta):
        if current_board.terminal() or depth == 0:
            # We must evaluate from the root‐MAX’s perspective, but here 
            # we’re calling evaluate(current_player, …), which passes the 
            # subnode’s “current_player” instead of the original `player`.
            #
            # CORRECT CALL: 
            return evaluate(max_player, current_board)

        if current_player == max_player:
            return max_value(current_player, current_board, depth, alpha, beta)
        else:
            return min_value(current_player, current_board, depth, alpha, beta)

    def max_value(current_player, current_board, depth, alpha, beta):
        v = -math.inf
        for col in range(current_board.cols):
            if not current_board.placeable(col):
                continue
            child = current_board.clone()
            child.place(current_player, col)

            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1, alpha, beta)
            if child_val > v:
                v = child_val

            # Update alpha
            if v > alpha:
                alpha = v

            # If α ≥ β, prune the rest
            if alpha >= beta:
                break
        return v

    def min_value(current_player, current_board, depth, alpha, beta):
        v = math.inf
        for col in range(current_board.cols):
            if not current_board.placeable(col):
                continue
            child = current_board.clone()
            child.place(current_player, col)

            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1, alpha, beta)
            if child_val < v:
                v = child_val

            # Update beta
            if v < beta:
                beta = v

            # If β ≤ α, prune the rest
            if beta <= alpha:
                break
        return v

    # If the board is already terminal → no move possible
    if board.terminal():
        return None

    # Root is a MAX node:
    alpha = -math.inf
    beta = math.inf
    best_score = -math.inf

    for col in range(board.cols):
        if not board.placeable(col):
            continue
        child = board.clone()
        child.place(player, col)

        next_p = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
        child_score = value(next_p, child, depth_limit - 1, alpha, beta)

        if child_score > best_score:
            best_score = child_score
            placement = col

        # Update alpha at root
        if child_score > alpha:
            alpha = child_score

        # If α ≥ β at root, prune siblings
        if alpha >= beta:
            break

    return placement




def expectimax(player, board, depth_limit):
    """
    Expectimax algorithm, where the “adversary” (other player) is modeled
    as choosing moves uniformly at random.

    Parameters
    ----------
    player : Board.PLAYER1 or Board.PLAYER2
        The “max” player whose turn it is at the root.
    board : Board
        The current game board.
    depth_limit : int
        How many plies to search before calling evaluate().

    Returns
    -------
    placement : int or None
        The column (0..cols-1) where `player` should drop a disk, or None if
        board is terminal.
    """
    max_player = player
    placement = None

    def value(current_player, current_board, depth):
        # Base case: terminal or depth == 0 → evaluate immediately
        if current_board.terminal() or depth == 0:
            # ✖ WRONG: this evaluates from current_player’s perspective
            #     return evaluate(current_player, current_board)
            #
            # CORRECT:
            return evaluate(max_player, current_board)

        if current_player == max_player:
            return max_value(current_player, current_board, depth)
        else:
            return chance_value(current_player, current_board, depth)

    def max_value(current_player, current_board, depth):
        v = -math.inf
        for col in range(current_board.cols):
            if not current_board.placeable(col):
                continue
            child = current_board.clone()
            child.place(current_player, col)

            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1)
            if child_val > v:
                v = child_val
        return v

    def chance_value(current_player, current_board, depth):
        """
        “Chance” node: opponent’s moves are chosen uniformly at random.
        We return the average (expected) value over all legal child boards.
        """
        legal_cols = [c for c in range(current_board.cols) if current_board.placeable(c)]
        if not legal_cols:   
            # CORRECT VERSION:
            return evaluate(max_player, current_board)

        total = 0.0
        for col in legal_cols:
            child = current_board.clone()
            child.place(current_player, col)

            next_p = (
                current_board.PLAYER2
                if current_player == current_board.PLAYER1
                else current_board.PLAYER1
            )
            child_val = value(next_p, child, depth - 1)
            total += child_val

        return total / len(legal_cols)

    # If the board is already terminal → no move possible
    if board.terminal():
        return None

    # Root is a MAX node (player to move is max_player)
    best_score = -math.inf
    for col in range(board.cols):
        if not board.placeable(col):
            continue
        child = board.clone()
        child.place(player, col)

        next_p = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
        child_score = value(next_p, child, depth_limit - 1)

        if child_score > best_score:
            best_score = child_score
            placement = col

    return placement




if __name__ == "__main__":
    from utils.app import App
    import tkinter

    algs = {
        "Minimax": minimax,
        "Alpha-beta pruning": alphabeta,
        "Expectimax": expectimax
    }

    root = tkinter.Tk()
    App(algs, root)
    root.mainloop()
