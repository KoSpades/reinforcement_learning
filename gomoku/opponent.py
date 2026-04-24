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

    def __init__(self, model_path):
        model_path = Path(model_path) 
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy = PolicyNetwork().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def select_action(self, state, whose_turn):
        """
        whose_turn: 0 if I am black, 1 if I am white.
        """
        if whose_turn == 0:
            policy_state = state
        else:
            policy_state = torch.stack([state[1], state[0]])

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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy = PolicyNetwork().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
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
    
