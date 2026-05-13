import torch
import math
from pathlib import Path

from config import BOARD_SIZE, ACTOR_CRITIC_MODELS_DIR
from model import PolicyNetwork
from utils import check_win_cond, get_policy_state, step


class Node:

    def __init__(self, whose_turn, P, N, W, parent, action, children):
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
        self.N = N
        self.W = W
        self.action = action
        self.parent = parent
        self.children = children
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

    def __init__(self, whose_turn, P, N, W, parent, action, state, children):
        super().__init__(whose_turn, P, N, W, parent, action, children)
        self.state = state


class MCTS:
    '''
    state: the initial board state
    whose_turn: whose turn is it to take the next move from the initial board state
    last_action: the last action that has led to the initial board state
    policy: a PolicyNetwork that outputs a distribution over actions and a value
    exploration_coef: c in PUCT formula
    total_sim_for_one_move: how many total simulations do we want
    '''

    def __init__(self, 
                 state,
                 whose_turn,
                 last_action,
                 policy: PolicyNetwork, 
                 exploration_coef=1, 
                 total_sim_for_one_move=20,
                 dirichlet_enabled=True,
                 dirichlet_alpha=0.2,
                 dirichlet_eps=0.25):
        self.root = Root(whose_turn=whose_turn, 
                         P=0, 
                         N=0, 
                         W=0, 
                         parent=None, 
                         action=last_action, 
                         state=state, 
                         children={})
        self.policy = policy
        self.device = next(self.policy.parameters()).device
        self.exploration_coef = exploration_coef
        self.total_sim_for_one_move=total_sim_for_one_move
        self.dirichlet_enabled = dirichlet_enabled
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def add_dirichlet_noise(self):
        """
        Adding Dirichlet noise to root node once. 
        It should be applied to the Root node, if the Root node has children, and dirichlet_enabled is True.
        """
        if not self.dirichlet_enabled or not self.root.children:
            return
        
        actions = list(self.root.children.keys())
        noise = torch.distributions.Dirichlet(
          torch.full((len(actions),), self.dirichlet_alpha, device=self.device)
        ).sample()
        for action, eta in zip(actions, noise):
            child = self.root.children[action]
            child.P = (1 - self.dirichlet_eps) * child.P + self.dirichlet_eps * eta.item()

    def select_action(self, return_stats=False):
        num_sim = 0
        cur_node = self.root

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

        while(num_sim <= self.total_sim_for_one_move):

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
                    cur_node = self.root
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
                    cur_node = self.root
                    continue

                # Case 2: Game has not ended yet: we do a value propagate based on NN outputed value
                with torch.no_grad():
                    policy_state = get_policy_state(cur_node_state, cur_node.whose_turn)
                    action_logits, cur_value = self.policy(policy_state.unsqueeze(0).to(self.device))
                    action_logits = action_logits.squeeze(0)
                    cur_value = cur_value.squeeze()
                    occupied_spaces = cur_node_state.sum(dim=0)
                    legal_mask = (occupied_spaces == 0).flatten().to(self.device)
                    masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
                    action_dist = torch.distributions.Categorical(logits=masked_logits)
                    legal_actions = legal_mask.nonzero().flatten().tolist()
                    for action in legal_actions:
                        child_node = Node(whose_turn=1 - cur_node.whose_turn, 
                                          P=action_dist.probs[action].item(),
                                          N=0,
                                          W=0,
                                          parent=cur_node,
                                          action=action,
                                          children={})
                        cur_node.children[action] = child_node
                    if cur_node.parent is None:
                        self.add_dirichlet_noise()
                    # back propagate the value from NN from current node upwards
                    cur_node.value_propagate(cur_value.item())
                    num_sim += 1
                    cur_node = self.root

            # Second Big Case: non-leaf Node traversal (we have seen this Node before, and are just picking a branch)
            else:
                PUCT_scores = {}
                for action, child in cur_node.children.items():
                    PUCT_scores[action] = -child.Q + self.exploration_coef * child.P * (math.sqrt(cur_node.N) / (1 + child.N))
                best_branch = max(PUCT_scores, key=PUCT_scores.get)
                cur_node = cur_node.children[best_branch]

        # Finally, choose the action corresponding to Root's children with the highest count N
        best_action = max(self.root.children,
                          key=lambda action: self.root.children[action].N)
        
        if return_stats:
            root_info = {
                action: {
                    "N": child.N,
                    "Q": -child.Q,
                    "P": child.P,
                    "exploration": math.sqrt(self.root.N) / (1 + child.N),
                    "prior_boost": child.P * (math.sqrt(self.root.N) / (1 + child.N)),
                }
                for action, child in self.root.children.items()
            }
        else:
            root_info = None
        
        # Update Root node for subtree reuse. Need to create a new Root object.
        new_root_state = step(self.root.state, best_action, self.root.whose_turn)
        self.root = Root(whose_turn=1-self.root.whose_turn, 
                         P=0, 
                         N=self.root.children[best_action].N,
                         W=self.root.children[best_action].W,
                         parent=None, 
                         action=best_action, 
                         state=new_root_state, 
                         children=self.root.children[best_action].children)
        
        for _, child in self.root.children.items():
            child.parent = self.root

        self.add_dirichlet_noise()

        return best_action, root_info
    
    def advance_root(self, action):
        '''
        action: an incoming action that would lead to a board change.
        We need to update the self.root accordingly.
        Note: player that took this action is the same as Root's whose_turn.
        '''
        # Two cases: if action is in Root's children, we reuse. Else start a fresh root node.
        new_root_state = step(self.root.state, action, self.root.whose_turn)
        if action in self.root.children:
            self.root = Root(whose_turn=1-self.root.whose_turn, 
                             P=0, 
                             N=self.root.children[action].N,
                             W=self.root.children[action].W,
                             parent=None, 
                             action=action, 
                             state=new_root_state, 
                             children=self.root.children[action].children)
        
            for _, child in self.root.children.items():
                child.parent = self.root
            self.add_dirichlet_noise()
        else:
            self.root = Root(whose_turn=1-self.root.whose_turn, 
                             P=0, 
                             N=0,
                             W=0,
                             parent=None, 
                             action=action, 
                             state=new_root_state, 
                             children={})


if __name__ == "__main__":
    device = "cpu"
    cur_state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), device=device)
    policy = PolicyNetwork().to(device)
    model_path = Path(ACTOR_CRITIC_MODELS_DIR / "final_policy_10000.pt")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"]
    policy.load_state_dict(state_dict)
    policy.eval()
    mcts = MCTS(state=cur_state,
                whose_turn=0,
                last_action=-1,
                policy=policy)
    for _ in range(2):
        print(mcts.select_action())
