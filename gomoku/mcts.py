import torch
import math
from pathlib import Path

from config import BOARD_SIZE, ACTOR_CRITIC_MODELS_DIR
from model import PolicyNetwork
from utils import check_win_cond, step

class Node:

    def __init__(self, whose_turn, P, parent, action):
        '''
        whose_turn: whose turn is it to place the NEXT MOVE.
        P: predicted likelihood of taking the action that led to this Node. Computed from NN.
        N: total number of visits.
        W: total value from simulation updates that have passed through this node.
        action: which action led to this Node from parent (0 to BOARD_SIZE**2-1)
        parent: pointer to parent
        childen: a map of (action, Child_Node).
        is_terminal: Bool
        winner: 0, 1, 2, or -1. It's -1 if is_terminal is False.
        '''
        self.whose_turn = whose_turn
        self.P = P
        self.N = 0
        self.W = 0
        self.action = action
        self.parent = parent
        self.children = {}
        self.is_terminal = False
        self.winner = -1

    @property
    def Q(self):
        if self.N:
            return self.W / self.N
        return 0
    
    @property
    def is_leaf(self):
        return not self.children

    def value_propagate(self, value):
        cur_node = self
        while cur_node is not None:
            cur_node.N += 1
            cur_node.W += value
            value = -value
            cur_node = cur_node.parent


class Root(Node):

    def __init__(self, whose_turn, P, parent, action, state):
        super().__init__(whose_turn, P, parent, action)
        self.state = state


def mcts_action_selection(state, whose_turn, last_action, policy: PolicyNetwork, exploration_coef=1, total_sim=20):
    '''
    state: the current board state
    whose_turn: whose turn is it to take the next move
    last_action: the last action that has led to this board
    policy: a PolicyNetwork that outputs a distribution over actions and a value
    total_sim: how many total simulations do we want
    '''
    num_sim = 0
    device = next(policy.parameters()).device
    root = Root(whose_turn=whose_turn, P=0, parent=None, action=last_action, state=state)
    cur_node = root

    def get_policy_state(state, whose_turn):
        if whose_turn == 0:
            return state
        else:
            return torch.stack([state[1], state[0]])
        
    def get_state_for_cur_node(cur_node):
        actions = []
        while cur_node.parent:
            actions.append(cur_node.action)
            cur_node = cur_node.parent
        actions.reverse()
        cur_state = cur_node.state
        cur_whose_turn = cur_node.whose_turn
        for action in actions:
            cur_state = step(cur_state, action, cur_whose_turn)
            cur_whose_turn = 1 - cur_whose_turn
        return cur_state

    while(num_sim < total_sim):

        # First Big Case: leaf node expansion (i.e. Nodes with no children)
        if cur_node.is_leaf:
            # There are two cases when a Node has no children
            # 1. It has not been expanded yet, but game isn't over.
            # 2. Game is at terminal state.

            # It may be a terminal node already: in this case, we just want to propagate its actual value up
            # Need to get the right value to propagate for this leaf.
            if cur_node.is_terminal:
                last_turn = 1 - cur_node.whose_turn
                if cur_node.winner == 2:
                    back_value = 0 
                elif last_turn == cur_node.winner:
                    # Note: this is -1, because node value is from the perspective of player who needs to move NOW
                    back_value = -1
                else:
                    back_value = 1
                cur_node.value_propagate(back_value)
                num_sim += 1
                cur_node = root
                continue

            # Even if the leaf is not alraedy marked as terminal, it still may be a new terminal node, so we need to check for that.
            # Note: since we are only storing the board state at the root, 
            # we need to step from root all the way down to the current node 
            # to cover the state corresponding to this node
            cur_node_state = get_state_for_cur_node(cur_node)
            winner = check_win_cond(cur_node_state, 1 - cur_node.whose_turn, cur_node.action)
            # Case 1: It actually is a new terminal node that we've just reached
            # We need to set the right status for the current node, then back propagate
            if winner > -1:
                cur_node.is_terminal = True
                cur_node.winner = winner
                if cur_node.winner == 2:
                    back_value = 0
                elif 1 - cur_node.whose_turn == winner:
                    back_value = -1
                else:
                    back_value = 1
                cur_node.value_propagate(back_value)
                num_sim += 1
                cur_node = root
                continue

            # Case 2: Game has not ended yet: we do a value propagate based on NN outputed value
            with torch.no_grad():
                policy_state = get_policy_state(cur_node_state, cur_node.whose_turn)
                action_logits, cur_value = policy(policy_state.unsqueeze(0).to(device))
                action_logits = action_logits.squeeze(0)
                cur_value = cur_value.squeeze()
                occupied_spaces = cur_node_state.sum(dim=0)
                legal_mask = (occupied_spaces == 0).flatten().to(device)
                masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
                action_dist = torch.distributions.Categorical(logits=masked_logits)
                legal_actions = legal_mask.nonzero().flatten().tolist()
                for action in legal_actions:
                    child_node = Node(whose_turn=1 - cur_node.whose_turn, 
                                    P=action_dist.probs[action].item(),
                                    parent=cur_node,
                                    action=action)
                    cur_node.children[action] = child_node
                # back propagate the value from NN from current node upwards
                cur_node.value_propagate(cur_value.item())
                num_sim += 1
                cur_node = root

        # Second Big Case: non-leaf Node traversal (we have seen this Node before, and are just picking a branch)
        else:
            PUCT_scores = {}
            for action, child in cur_node.children.items():
                PUCT_scores[action] = -child.Q + exploration_coef * child.P * (math.sqrt(cur_node.N) / (1 + child.N))
            best_branch = max(PUCT_scores, key=PUCT_scores.get)
            cur_node = cur_node.children[best_branch]

    # Finally, choose the action corresponding to Root's children with the highest count N
    best_action = max(root.children,
                      key=lambda action: root.children[action].N)
    
    return best_action


if __name__ == "__main__":
    device = "cpu"
    cur_state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), device=device)
    policy = PolicyNetwork().to(device)
    model_path = Path(ACTOR_CRITIC_MODELS_DIR / "final_policy_10000.pt")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"]
    policy.load_state_dict(state_dict)
    policy.eval()
    mcts_action_selection(state=cur_state, 
                          whose_turn=0, 
                          last_action=-1,
                          policy=policy)
