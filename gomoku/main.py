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
from config import BOARD_SIZE, TRAIN_ITER, TRAINING_ALGO, PLOTS_DIR, MODEL_DIRS_BY_ALGO
    

def generate_episode_for_reinforce(start_state, self_play, our_player, opponent, random_start=True):
    """
    Return a complete episode of (state, action) pairs, and the final reward (who won). 
    Starting from start_state, following the input player objects.
    """
    episode_history = []
    black_log_probs = []
    white_log_probs = []
    trainable_entropies = []
    black_predicted = []
    white_predicted = []
    # cur_turn: 0 for black, 1 for white.
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
        # In this case, we are sampling our the(S, A)-pairs for training the NN
        # This happens under two scenarios: it's our own turn (rather than a fixed opponent), or if we are doing self-play
        if our_turn or self_play:
            active_player = our_player if our_turn else opponent
            cur_action, cur_action_log_prob, cur_entropy, cur_value = active_player.select_action(cur_state, cur_turn, sample=True)
            if cur_action == -1:
                # In this case, we got a draw, and there's nothing to train, so we just ignore this episode.
                episode_history = []
                black_log_probs = []
                white_log_probs = []
                trainable_entropies = []
                black_predicted = []
                white_predicted = []
                break
            episode_history.append((cur_state, cur_action))
            if cur_turn == 0:
                black_log_probs.append(cur_action_log_prob)
                black_predicted.append(cur_value)
            else:
                white_log_probs.append(cur_action_log_prob)
                white_predicted.append(cur_value)
            trainable_entropies.append(cur_entropy)
        # In this case: the "action" is just for rolling out the environment, not used for updating model weights.
        else:
            cur_action = opponent.select_action(cur_state, cur_turn)
            if cur_action == -1:
                episode_history = []
                black_log_probs = []
                white_log_probs = []
                trainable_entropies = []
                black_predicted = []
                white_predicted = []
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
        "black_predicted": black_predicted,
        "white_predicted": white_predicted,
        "our_color": our_color,
    }


def compute_total_losses(episode_data, self_play, training_algo, regular_beta, device):
    winner = episode_data["episode_history"][-1][1]
    black_log_probs = episode_data["black_log_probs"]
    white_log_probs = episode_data["white_log_probs"]
    black_predicted = episode_data["black_predicted"]
    white_predicted = episode_data["white_predicted"]

    # Case 1: self-play
    # both black and white side should get a reward
    if self_play:
        policy_reward_black = 1 if winner == 0 else -1
        policy_reward_white = -policy_reward_black
    # Case 2: fixed opponent
    # we shall only use OUR reward
    else:
        our_color = episode_data["our_color"]
        our_reward = 1 if winner == our_color else -1
        policy_reward_black = our_reward if our_color == 0 else 0
        policy_reward_white = our_reward if our_color == 1 else 0

    # First: calculate the actor loss

    if len(black_log_probs) and training_algo == "reinforce":
        black_log_probs = torch.stack(black_log_probs)
        black_actor_loss = -black_log_probs.mean() * policy_reward_black
    elif len(black_log_probs) and training_algo == "actor_critic":
        black_log_probs = torch.stack(black_log_probs)
        black_predicted = torch.stack(black_predicted)
        black_advantage = policy_reward_black - black_predicted.detach()
        black_actor_loss = -(black_log_probs * black_advantage).mean()
    else:
        black_actor_loss = torch.tensor(0.0, device=device)

    if len(white_log_probs) and training_algo == "reinforce":
        white_log_probs = torch.stack(white_log_probs)
        white_actor_loss = -white_log_probs.mean() * policy_reward_white
    elif len(white_log_probs) and training_algo == "actor_critic":
        white_log_probs = torch.stack(white_log_probs)
        white_predicted = torch.stack(white_predicted)
        white_advantage = policy_reward_white - white_predicted.detach()
        white_actor_loss = -(white_log_probs * white_advantage).mean()
    else:
        white_actor_loss = torch.tensor(0.0, device=device)

    actor_loss = black_actor_loss + white_actor_loss

    # Second: calculate the critic loss
    if len(black_predicted) and training_algo == "actor_critic":
        if not torch.is_tensor(black_predicted):
            black_predicted = torch.stack(black_predicted)
        black_critic_loss = ((black_predicted - policy_reward_black)**2).mean()
    else:
        black_critic_loss = torch.tensor(0.0, device=device)

    if len(white_predicted) and training_algo == "actor_critic":
        if not torch.is_tensor(white_predicted):
            white_predicted = torch.stack(white_predicted)
        white_critic_loss = ((white_predicted - policy_reward_white)**2).mean()
    else:
        white_critic_loss = torch.tensor(0.0, device=device)

    critic_loss = black_critic_loss + white_critic_loss

    # Finally: calculate the entropy loss
    trainable_entropies = episode_data["trainable_entropies"]
    if len(trainable_entropies):
        entropy_loss = -regular_beta * torch.stack(trainable_entropies).mean()
    else:
        entropy_loss = torch.tensor(0.0, device=device)

    return actor_loss, critic_loss, entropy_loss


def rl_training_loop(start_state, 
                     self_play,
                     training_algo,
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
    cur_policy = PolicyNetwork().to(start_state.device)
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
            actor_loss, critic_loss, entropy_loss = compute_total_losses(
                episode_data=episode_data,
                self_play=self_play,
                training_algo=training_algo,
                regular_beta=regular_beta,
                device=start_state.device,
            )
            if training_algo == "reinforce":
                unregularized_loss = actor_loss
                total_loss = actor_loss + entropy_loss
            elif training_algo == "actor_critic":
                unregularized_loss = actor_loss + critic_loss
                total_loss = actor_loss + critic_loss + entropy_loss
            else:
                raise ValueError(f"Training algorithm {training_algo} not implemented yet!")
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_by_iter.append(unregularized_loss.detach())
            cur_iter += 1

    # Saving the model and optimizer for later, cotinuous training if needed
    model_dir = MODEL_DIRS_BY_ALGO[training_algo]
    model_dir.mkdir(parents=True, exist_ok=True)

    if player_path is None:
        torch.save({
            "model": cur_policy.state_dict(),
            "optimizer": optimizer.state_dict()
        }, model_dir / f"final_policy_{TRAIN_ITER}.pt")
        print(f"Training completed, saved to new path {model_dir / f'final_policy_{TRAIN_ITER}.pt'}")
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
    # my_opponent = FirstOpponent()
    # final_policy, loss_list = rl_training_loop(cur_state, 
    #                                          self_play=False, 
    #                                          training_algo=TRAINING_ALGO,
    #                                          player_path=MODEL_DIRS_BY_ALGO[TRAINING_ALGO] / "final_policy_10000.pt", 
    #                                          opponent=my_opponent,
    #                                          num_iter=TRAIN_ITER)
    final_policy, loss_list = rl_training_loop(cur_state, 
                                               self_play=True, 
                                               training_algo=TRAINING_ALGO,
                                               num_iter=TRAIN_ITER)
    print(f"Total training time is {time.time() - start_time}")

    plot_loss_by_iter(loss_list=loss_list)
