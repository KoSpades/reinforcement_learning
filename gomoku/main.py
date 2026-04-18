import torch

def move_is_legal(state, action):
    """
    Checking whether a move is legal. Output True if yes.
    Logic:
        Just checks if the position is occupied by neither black nor white by inspecting the states
    """
    if 0 <= action <= 80:
        row_idx = action // 9
        col_idx = action % 9
        return (state[0][row_idx][col_idx] == 0) and (state[1][row_idx][col_idx] == 0)
    return False

def step(state, action, whose_turn):
    """
    INPUT: 
        state: 
            a (2*9*9) representing pieces on the board.
            First dimension corresponds to the black pieces: 1 if occupied, 0 o/w.
            Second dimension corresponds to the white pieces.
        action:
            an integer from 0 to 80 
        whose_turn:
            0 if black, 1 if white.
    OUTPUT:
        the next state
    Note:
        We also should check for move validity -> this shall have an impact later on
    """
    if move_is_legal(state, action):
        row_idx = action // 9
        col_idx = action % 9
        state[whose_turn][row_idx][col_idx] = 1
        return state
    else:
        raise ValueError(f"Illegal move at action {action}")