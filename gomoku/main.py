import torch
import time
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from config import BOARD_SIZE, TRAIN_ITER


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


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers =nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        )


    def forward(self, x):
        logits = self.layers(x)
        return logits
    

def generate_episode_for_reinforce(start_state, policy):
    """
    Return a complete episode of (state, action) pairs, and the final reward (who won). 
    Starting from start_state, following the input policy parameter.
    """
    episode_history = []
    action_log_probs = []
    cur_state = start_state
    cur_turn = 0
    while (True):
        # generate an action using the NN
        if cur_turn == 0:
            policy_state = cur_state
        else:
            policy_state = torch.stack([cur_state[1], cur_state[0]])
        action_logits = policy(policy_state.unsqueeze(0)).squeeze(0)
        occupied_spaces = cur_state.sum(dim=0)
        legal_mask = (occupied_spaces == 0).flatten()
        if not legal_mask.any():
            episode_history = []
            action_log_probs = []
            break
        masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
        action_probs = F.softmax(masked_logits, dim=-1) # Note: dim=-1 since there's a batch dimension in the front
        action_dist = torch.distributions.Categorical(probs=action_probs)
        cur_action = action_dist.sample()
        cur_action_log_prob = action_dist.log_prob(cur_action)
        episode_history.append((cur_state, cur_action))
        action_log_probs.append(cur_action_log_prob)
        next_state = step(cur_state, cur_action, cur_turn)
        # Checking for termination
        res = check_win_cond(next_state, cur_turn, cur_action)
        # First case: not terminated yet
        if res < 0:
            cur_state = next_state
            cur_turn = 1 - cur_turn
        # Second case: reached termination
        else:
            if res == 0:
                episode_history.append((next_state, 1))
            else:
                episode_history.append((next_state, -1))
            break
    return episode_history, action_log_probs
    

def reinforce_algo(start_state, num_iter=1000):
    cur_iter = 0
    cur_policy = PolicyNetwork().to(start_state.device)
    optimizer = torch.optim.Adam(cur_policy.parameters(), lr=1e-3)
    loss_by_iter = []
    while (cur_iter < num_iter):
        if (cur_iter % 100 == 0):
            print(f"Current iter: {cur_iter}")
        # generate a complete episode
        cur_episode, cur_actions_probs = generate_episode_for_reinforce(start_state, cur_policy)
        # pretty_print_state(cur_episode[-1][0])

        # First, split the episode into two trajectories since we are doing self-play.
        if len(cur_episode):
            black_reward = cur_episode[-1][1]
            white_reward = -black_reward
            cur_episode = cur_episode[:-1]
            # black_episode = cur_episode[::2]
            # white_episode = cur_episode[1::2]
            black_log_probs = cur_actions_probs[::2]
            white_log_probs = cur_actions_probs[1::2]

            # Weight updates for both trajectories
            black_loss = -torch.stack(black_log_probs).sum() * black_reward
            white_loss = -torch.stack(white_log_probs).sum() * white_reward
            total_loss = black_loss + white_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_by_iter.append(total_loss)
            cur_iter += 1

    return cur_policy, loss_by_iter


def plot_loss_by_iter(loss_list):
    loss_list = [loss.item() if torch.is_tensor(loss) else loss for loss in loss_list]
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(output_dir / f"loss_curve_{TRAIN_ITER}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    print("**************Training start**************")
    start_time = time.time()
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    cur_state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), device=device)
    final_policy, loss_list = reinforce_algo(cur_state, TRAIN_ITER)
    output_dir = Path(__file__).resolve().parent
    torch.save(final_policy.state_dict(), output_dir / f"final_policy_{TRAIN_ITER}.pt")
    print(f"Total training time is {time.time() - start_time}")

    plot_loss_by_iter(loss_list=loss_list)
