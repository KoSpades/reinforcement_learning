from pathlib import Path

import torch

from config import MODELS_DIR
from model import PolicyNetwork
from utils import check_win_cond, get_random_legal_move, step


class Player:
    pass


class OurPlayer(Player):
    """
    Our player: who acts strictly according to a NN.
    """

    def __init__(self, model_path=None, policy=None):
        if policy is not None:
            self.policy = policy
            self.device = next(self.policy.parameters()).device
        else:
            self.device = torch.device("cpu")
            self.policy = PolicyNetwork().to(self.device)
            if model_path is not None:
                model_path = Path(model_path)
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                self.policy.load_state_dict(state_dict)
            self.policy.eval()

    def _policy_state(self, state, whose_turn):
        if whose_turn == 0:
            return state
        return torch.stack([state[1], state[0]])

    def select_action(self, state, whose_turn, sample=False):
        """
        whose_turn: 0 if I am black, 1 if I am white.
        """
        policy_state = self._policy_state(state, whose_turn)
        # We randomly sample an action
        if sample:
            action_logits = self.policy(policy_state.unsqueeze(0).to(self.device)).squeeze(0)
            occupied_spaces = state.sum(dim=0)
            legal_mask = (occupied_spaces == 0).flatten().to(self.device)
            if not legal_mask.any():
                return -1, None, None
            masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
            action_dist = torch.distributions.Categorical(logits=masked_logits)
            cur_action = action_dist.sample()
            cur_action_log_prob = action_dist.log_prob(cur_action)
            cur_entropy = action_dist.entropy()
            return int(cur_action.item()), cur_action_log_prob, cur_entropy
        # We deterministically pick an action (used in evaluation)
        else:
            with torch.no_grad():
                action_logits = self.policy(policy_state.unsqueeze(0).to(self.device)).squeeze(0)
                occupied_spaces = state.sum(dim=0)
                legal_mask = (occupied_spaces == 0).flatten().to(self.device)
                if not legal_mask.any():
                    return -1
                masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
                return int(torch.argmax(masked_logits).item())



class RandomOpponent(Player):
    """
    The most basic opponent: chooses random moves.
    """

    def select_action(self, state, whose_turn=None):
        if not (state.sum(dim=0) == 0).any():
            return -1
        return get_random_legal_move(state)


class FirstOpponent(Player):
    """
    Opponent number 1! With the following heuristic:
    - If it can win, it wins.
    - Elif: opponent has 4 in a row, it blocks it.
    - Else: we use the weights from a freezed NN (models/freeze_10000_zero.pt) to pick a move.

    The goal is to train against this opponent, and we will get higher and higher quality freezed NN's.
    """

    def __init__(self, model_path=None):
        model_path = Path(model_path) if model_path is not None else MODELS_DIR / "freeze_10000_zero.pt"
        self.device = torch.device("cpu")
        self.policy = PolicyNetwork().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def _legal_actions(self, state):
        occupied = state.sum(dim=0)
        return (occupied == 0).flatten().nonzero().flatten()

    def _find_immediate_move(self, state, whose_turn):
        for action in self._legal_actions(state):
            next_state = step(state, action, whose_turn)
            if check_win_cond(next_state, whose_turn, action) == whose_turn:
                return int(action.item())
        return None

    def _policy_action(self, state, whose_turn):
        if whose_turn == 0:
            policy_state = state
        else:
            policy_state = torch.stack([state[1], state[0]])

        with torch.no_grad():
            action_logits = self.policy(policy_state.unsqueeze(0).to(self.device)).squeeze(0)
            occupied_spaces = state.sum(dim=0)
            legal_mask = (occupied_spaces == 0).flatten().to(self.device)
            masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
            return int(torch.argmax(masked_logits).item())

    def select_action(self, state, whose_turn):
        """
        whose_turn: 0 if I am black, 1 if I am white.
        """
        if not (state.sum(dim=0) == 0).any():
            return -1
        winning_action = self._find_immediate_move(state, whose_turn)
        if winning_action is not None:
            return winning_action

        blocking_action = self._find_immediate_move(state, 1 - whose_turn)
        if blocking_action is not None:
            return blocking_action

        return self._policy_action(state, whose_turn)
    
