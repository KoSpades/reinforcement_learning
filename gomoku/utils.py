import torch

from config import BOARD_SIZE


def move_is_legal(state, action):
    """
    Checking whether a move is legal. Output True if yes.
    Logic:
        Just checks if the position is occupied by neither black nor white by inspecting the states
    """
    if 0 <= action <= BOARD_SIZE**2 - 1:
        row_idx = action // BOARD_SIZE
        col_idx = action % BOARD_SIZE
        return (state[0][row_idx][col_idx] == 0) and (state[1][row_idx][col_idx] == 0)
    return False


def get_random_legal_move(state):
    """
    Output an integer for the position of a valid move.
    """
    # First, combine the black and white place
    occupied = state.sum(dim=0)
    # Then, construct a boolean mask for legal positions and flatten it
    legal_mask = (occupied == 0)
    legal_flat = legal_mask.flatten()
    # Then, get the legal move indices and flatten it
    legal_actions = legal_flat.nonzero().flatten()
    # Finally, pick a random idx and get its value
    legal_idx = torch.randint(len(legal_actions), ())
    return legal_actions[legal_idx]


def check_win_cond(state, whose_turn, action):
    """
    Input:
        state
        whose_turn: whoever just made a move (e.g. if black just made a move, whose_turn is 0)
        action: what's the last move that led to the current state
    Output:
        0 if black wins, 1 if white. -1 o/w
    """ 
    def count_direction(board, row, col, row_dir, col_dir):
        count = 1
        r, c = row + row_dir, col + col_dir
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 1:
            count += 1
            r += row_dir
            c += col_dir
        r, c = row - row_dir, col - col_dir
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 1:
            count += 1
            r -= row_dir
            c -= col_dir
        return count
        
    # Set up the shape checks: We need four shapes: horizontal, diagonal, and two diagonal
    action = int(action)
    board = state[whose_turn]

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    for dr, dc in directions:
        if count_direction(board, row, col, dr, dc) >= 5:
            return whose_turn
        
    return -1


def step(state, action, whose_turn):
    """
    INPUT: 
        state: 
            a (2*BOARD_SIZE*BOARD_SIZE) tensor representing pieces on the board.
            First dimension corresponds to the black pieces: 1 if occupied, 0 o/w.
            Second dimension corresponds to the white pieces.
        action:
            an integer from 0 to (BOARD_SIZE**2 - 1)
        whose_turn:
            0 if black, 1 if white.
    OUTPUT:
        the next state
    Note:
        We also should check for move validity -> this shall have an impact later on
    """
    if move_is_legal(state, action):
        row_idx = action // BOARD_SIZE
        col_idx = action % BOARD_SIZE
        next_state = state.clone()
        next_state[whose_turn][row_idx][col_idx] = 1
        return next_state
    else:
        raise ValueError(f"Illegal move at action {action}")
    

def pretty_print_state(state):
    """
    Pretty-print a Gomoku board.

    state shape:
        (2, BOARD_SIZE, BOARD_SIZE)

    state[0] = black stones
    state[1] = white stones
    """
    symbols = {
        0: ".",
        1: "B",
        2: "W",
    }

    occupied = state[0] + 2 * state[1]

    print("   " + " ".join(str(i) for i in range(BOARD_SIZE)))

    for row_idx in range(BOARD_SIZE):
        row_symbols = []
        for col_idx in range(BOARD_SIZE):
            cell = int(occupied[row_idx][col_idx].item())
            row_symbols.append(symbols[cell])

        print(f"{row_idx:2} " + " ".join(row_symbols))