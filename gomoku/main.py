import torch
import time
from torch import nn
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

from utils import check_win_cond, get_random_legal_move, step
from model import PolicyNetwork
from config import BOARD_SIZE, TRAIN_ITER, MODELS_DIR, PLOTS_DIR
    

def generate_episode_for_reinforce(start_state, policy, random_start=True):
    """
    Return a complete episode of (state, action) pairs, and the final reward (who won). 
    Starting from start_state, following the input policy parameter.
    Self-play: both players use the same policy NN.
    """
    episode_history = []
    action_log_probs = []
    entropies = []
    cur_turn = 0
    if random_start:
        total_random_moves = random.randint(2,  2)
        for _ in range(total_random_moves):
            cur_action = get_random_legal_move(start_state)
            start_state = step(start_state, cur_action, cur_turn)
            cur_turn = 1 - cur_turn
    cur_state = start_state
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
            entropies = []
            break
        masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
        action_probs = F.softmax(masked_logits, dim=-1) # Note: dim=-1 since there's a batch dimension in the front
        action_dist = torch.distributions.Categorical(probs=action_probs)
        cur_action = action_dist.sample()
        cur_action_log_prob = action_dist.log_prob(cur_action)
        cur_entropy = action_dist.entropy()
        episode_history.append((cur_state, cur_action))
        action_log_probs.append(cur_action_log_prob)
        entropies.append(cur_entropy)
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
    return episode_history, action_log_probs, entropies
    

def reinforce_algo(start_state, num_iter=1000, learning_rate=1e-3, regular_beta=0.01):
    cur_iter = 0
    cur_policy = PolicyNetwork().to(start_state.device)
    optimizer = torch.optim.Adam(cur_policy.parameters(), lr=learning_rate)
    loss_by_iter = []
    while (cur_iter < num_iter):
        if (cur_iter % 100 == 0):
            print(f"Current iter: {cur_iter}")
        # generate a complete episode
        cur_episode, cur_actions_probs, cur_entropies = generate_episode_for_reinforce(start_state, cur_policy)
        # pretty_print_state(cur_episode[-1][0])

        # First, split the episode into two trajectories since we are doing self-play.
        if len(cur_episode):
            black_reward = cur_episode[-1][1]
            white_reward = -black_reward
            cur_episode = cur_episode[:-1]
            black_log_probs = cur_actions_probs[::2]
            white_log_probs = cur_actions_probs[1::2]

            # Weight updates for both trajectories
            black_loss = -torch.stack(black_log_probs).sum() * black_reward
            white_loss = -torch.stack(white_log_probs).sum() * white_reward
            total_loss = black_loss + white_loss
            regularized_loss = total_loss - regular_beta * torch.stack(cur_entropies).mean() 
            optimizer.zero_grad()
            regularized_loss.backward()
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
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"loss_curve_{TRAIN_ITER}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    print("**************Training start**************")
    start_time = time.time()
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    cur_state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), device=device)
    final_policy, loss_list = reinforce_algo(cur_state, TRAIN_ITER)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(final_policy.state_dict(), MODELS_DIR / f"final_policy_{TRAIN_ITER}.pt")
    print(f"Total training time is {time.time() - start_time}")

    plot_loss_by_iter(loss_list=loss_list)
