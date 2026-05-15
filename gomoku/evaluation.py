from utils import get_random_legal_move, step, check_win_cond
from opponent import RandomOpponent, FirstOpponent, OurPlayer
from config import BOARD_SIZE, REINFORCE_MODELS_DIR, ACTOR_CRITIC_MODELS_DIR, MCTS_MODELS_DIR
from mcts import MCTS

import random
import torch


def calc_win_rate(our_player: OurPlayer, 
                  opponent, 
                  num_games=100, 
                  random_start=True,
                  use_mcts=False,
                  dirichlet_enabled=False):
    """
    use_mcts: when set of True, our_player will use MCTS (and our_player's NN) for action selection
    """
    device = "cpu"
    blank_state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), device=device)
    num_wins = 0
    num_draws = 0
    game_count = 0
    while(game_count < num_games):
        # Generate a random starting board position
        cur_state = blank_state
        cur_turn = 0
        if random_start:
            total_random_moves = random.randint(1,  2)
            for _ in range(total_random_moves):
                cur_action = get_random_legal_move(cur_state)
                cur_state = step(cur_state, cur_action, cur_turn)
                cur_turn = 1 - cur_turn
        else:
            cur_action = -1
        # Create an MCTS class if needed
        if use_mcts:
            mcts = MCTS(state=cur_state,
                        whose_turn=cur_turn,
                        last_action=cur_action,
                        policy=our_player.policy,
                        dirichlet_enabled=dirichlet_enabled,
                        total_sim_for_one_move=200)
        # Roll out a game, randomly choose black or white for our player
        our_color = random.randint(0, 1)
        our_turn = (cur_turn == our_color)
        while (True):
            # Generate an action for our player
            if our_turn:
                if use_mcts:
                    cur_action, _, _ = mcts.select_action()
                else:
                    cur_action = our_player.select_action(cur_state, cur_turn)
            else:
                cur_action = opponent.select_action(cur_state, cur_turn)
                if cur_action != -1 and use_mcts:
                    mcts.advance_root(cur_action)
            if cur_action == -1:
                num_draws += 1
                break
            next_state = step(cur_state, cur_action, cur_turn)
            res = check_win_cond(next_state, cur_turn, cur_action)
            # First case: not terminated yet
            if res < 0:
                cur_state = next_state
                cur_turn = 1 - cur_turn
                our_turn = not our_turn
            elif res == 2:
                num_draws += 1
                break
            # Second case: reached win termination
            else:
                num_wins += (res == our_color)
                break
        game_count += 1
    return num_wins, num_draws,num_games


if __name__ == "__main__":
    print("-----------Evaluation starting-------------")

    our_player = OurPlayer(MCTS_MODELS_DIR / "final_policy_1000.pt")

    # our_opponent = RandomOpponent()
    # num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True)
    # print(f"Win rate against random: {num_wins / num_games}")

    # our_opponent = OurPlayer(ACTOR_CRITIC_MODELS_DIR / "final_policy_1000.pt")
    # num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True)
    # print(f"Win rate against 1000 Actor-Critic: {num_wins / num_games}")

    our_opponent = FirstOpponent(ACTOR_CRITIC_MODELS_DIR / "final_policy_1000.pt")
    num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True, dirichlet_enabled=False)
    print(f"Win rate against Actor-Critic heuristic: {num_wins / num_games}")

    # our_opponent = OurPlayer(REINFORCE_MODELS_DIR / "final_policy_1000.pt")
    # num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True)
    # print(f"Win rate against 1000 REINFORCE: {num_wins / num_games}")

    our_opponent = FirstOpponent(REINFORCE_MODELS_DIR / "final_policy_1000.pt")
    num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True, dirichlet_enabled=False)
    print(f"Win rate against REINFORCE heuristic: {num_wins / num_games}")

    our_opponent = OurPlayer(ACTOR_CRITIC_MODELS_DIR / "final_policy_10000.pt")
    num_wins, num_draws, num_games = calc_win_rate(our_player, our_opponent, use_mcts=True, dirichlet_enabled=False)
    print(f"Win rate against 10000 AC: {num_wins / num_games}")
