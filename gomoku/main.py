import torch
import time
from torch import nn
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

from utils import check_win_cond, get_random_legal_move, step
from model import PolicyNetwork
from opponent import *
from config import BOARD_SIZE, TRAIN_ITER, MODELS_DIR, PLOTS_DIR
    

def generate_episode_for_reinforce(start_state, self_play, our_player, opponent, random_start=True):
    """
    Return a complete episode of (state, action) pairs, and the final reward (who won). 
    Starting from start_state, following the input player objects.
    """
    episode_history = []
    black_log_probs = []
    white_log_probs = []
    trainable_entropies = []
    cur_turn = 0
    if random_start:
        total_random_moves = random.randint(2,  2)
        for _ in range(total_random_moves):
            cur_action = get_random_legal_move(start_state)
            start_state = step(start_state, cur_action, cur_turn)
            cur_turn = 1 - cur_turn
    cur_state = start_state
    our_color = random.randint(0, 1)
    our_turn = (cur_turn == our_color)
    while (True):
        if our_turn or self_play:
            active_player = our_player if our_turn else opponent
            cur_action, cur_action_log_prob, cur_entropy = active_player.select_action(cur_state, cur_turn, sample=True)
            if cur_action == -1:
                # In this case, we got a draw, and there's nothing to train, so we just ignore this episode.
                episode_history = []
                black_log_probs = []
                white_log_probs = []
                trainable_entropies = []
                break
            episode_history.append((cur_state, cur_action))
            if cur_turn == 0:
                black_log_probs.append(cur_action_log_prob)
            else:
                white_log_probs.append(cur_action_log_prob)
            trainable_entropies.append(cur_entropy)
        else:
            cur_action = opponent.select_action(cur_state, cur_turn)
            if cur_action == -1:
                episode_history = []
                black_log_probs = []
                white_log_probs = []
                trainable_entropies = []
                break
        next_state = step(cur_state, cur_action, cur_turn)
        # Checking for termination
        res = check_win_cond(next_state, cur_turn, cur_action)
        # First case: not terminated yet
        if res < 0:
            cur_state = next_state
            cur_turn = 1 - cur_turn
            our_turn = not our_turn
        # Second case: reached termination
        else:
            episode_history.append((next_state, res))
            break
    return {
        "episode_history": episode_history,
        "black_log_probs": black_log_probs,
        "white_log_probs": white_log_probs,
        "trainable_entropies": trainable_entropies,
        "our_color": our_color,
    }


def compute_losses_for_reinforce(episode_data, self_play, regular_beta, device):
    winner = episode_data["episode_history"][-1][1]
    black_log_probs = episode_data["black_log_probs"]
    white_log_probs = episode_data["white_log_probs"]

    if self_play:
        policy_reward_black = 1 if winner == 0 else -1
        policy_reward_white = -policy_reward_black
    else:
        our_color = episode_data["our_color"]
        our_reward = 1 if winner == our_color else -1
        policy_reward_black = our_reward if our_color == 0 else 0
        policy_reward_white = our_reward if our_color == 1 else 0

    if len(black_log_probs):
        black_loss = -torch.stack(black_log_probs).mean() * policy_reward_black
    else:
        black_loss = torch.tensor(0.0, device=device)

    if len(white_log_probs):
        white_loss = -torch.stack(white_log_probs).mean() * policy_reward_white
    else:
        white_loss = torch.tensor(0.0, device=device)

    total_loss = black_loss + white_loss
    trainable_entropies = episode_data["trainable_entropies"]
    if len(trainable_entropies):
        regularized_loss = total_loss - regular_beta * torch.stack(trainable_entropies).mean()
    else:
        regularized_loss = total_loss

    return total_loss, regularized_loss


def reinforce_algo(start_state, 
                   self_play,
                   player_path=None,
                   opponent=None,
                   num_iter=1000, 
                   learning_rate=1e-3, 
                   regular_beta=0.01):
    """
    Only when self_play is True do we want to train the opponent as well.
    """
    if not self_play and opponent is None:
        raise ValueError("Not in self-play mode: must create an opponent.")
    cur_policy, _ = PolicyNetwork().to(start_state.device)
    optimizer = torch.optim.Adam(cur_policy.parameters(), lr=learning_rate)
    cur_iter = 0
    if player_path is not None:
        check_point = torch.load(player_path, map_location=start_state.device)
        cur_policy.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
    our_player = OurPlayer(policy=cur_policy)
    if self_play:
        opponent = OurPlayer(policy=cur_policy)
    loss_by_iter = []
    while (cur_iter < num_iter):
        if (cur_iter % 100 == 0):
            print(f"Current iter: {cur_iter}")
        # generate a complete episode
        episode_data = generate_episode_for_reinforce(start_state, self_play, our_player, opponent)
        cur_episode = episode_data["episode_history"]

        # perform updates
        if len(cur_episode):
            total_loss, regularized_loss = compute_losses_for_reinforce(
                episode_data=episode_data,
                self_play=self_play,
                regular_beta=regular_beta,
                device=start_state.device,
            )
            optimizer.zero_grad()
            regularized_loss.backward()
            optimizer.step()

            loss_by_iter.append(total_loss)
            cur_iter += 1

    # Saving the model and optimizer for later, cotinuous training if needed
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if player_path is None:
        torch.save({
            "model": cur_policy.state_dict(),
            "optimizer": optimizer.state_dict()
        }, MODELS_DIR / f"final_policy_{TRAIN_ITER}.pt")
        print(f"Training completed, saved to new path final_policy_{TRAIN_ITER}.pt")
    else:
        torch.save({
            "model": cur_policy.state_dict(),
            "optimizer": optimizer.state_dict()
        }, player_path)
        print(f"Training completed, saved to existing path {player_path}")

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
    my_opponent = FirstOpponent()
    # final_policy, loss_list = reinforce_algo(cur_state, 
    #                                          self_play=False, 
    #                                          player_path=MODELS_DIR / "final_policy_10000.pt", 
    #                                          opponent=my_opponent,
    #                                          num_iter=TRAIN_ITER)
    final_policy, loss_list = reinforce_algo(cur_state, 
                                             self_play=True, 
                                             num_iter=TRAIN_ITER)
    print(f"Total training time is {time.time() - start_time}")

    plot_loss_by_iter(loss_list=loss_list)
