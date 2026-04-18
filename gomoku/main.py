import torch
import torch.nn.functional as F


BOARD_SIZE = 9


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

def check_win_cond(state, whose_turn):
    """
    Input:
        state
        whose_turn: we will only be checking win cond after a specific player makes a move, so it's fine to include this
    Output:
        0 if black wins, 1 if white. -1 o/w
    """ 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Set up the shape checks: We need four shapes: horizontal, diagonal, and two diagonal
    kernels = [
        torch.ones((1, 1, 1, 5), device=device),
        torch.ones((1, 1, 5, 1), device=device),
        torch.eye(5, device=device).view(1, 1, 5, 5),
        torch.fliplr(torch.eye(5, device=device)).view(1, 1, 5, 5)
    ]
    player_board = state[whose_turn].view(1, 1, BOARD_SIZE, BOARD_SIZE)

    for ker in kernels:
        if (F.conv2d(player_board, ker) >= 5).any():
            return whose_turn
    return -1


def step(state, action, whose_turn):
    """
    INPUT: 
        state: 
            a (2*BOARD_SIZE*BOARD_SIZE) representing pieces on the board.
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
        state[whose_turn][row_idx][col_idx] = 1
        return state
    else:
        raise ValueError(f"Illegal move at action {action}")
    

if __name__ == "__main__":
    # check_win_cond()
    print("Game start")
    # device = device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # test_board = torch.ones((2, 9, 9), device=device)
    # test_board[0, 0, 0] = 0
    # test_board[1, 0, 0] = 0
    # test_board[0, 1, 0] = 0
    # test_board[1, 1, 0] = 0
    # print(get_random_legal_move(test_board))