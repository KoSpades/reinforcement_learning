from config import BOARD_SIZE
from utils import check_win_cond, get_random_legal_move, step

class Node:

    def __init__(self, whose_turn):
        '''
        whose_turn: whose turn is it to place the NEXT MOVE.
        P: predicted likelihood of taking the action that led to this Node. Computed from NN.
        N: total number of visits.
        W: total value from simulation updates that have passed through this node.
        action: which action led to this Node from parent (0 to BOARD_SIZE**2-1)
        parent: pointer to parent
        childen: a map of (action, Child_Node).
        is_terminal: Bool
        winner: 0, 1, or None. It's 0 or 1 if is_terminal is True.
        '''
        self.whose_turn = whose_turn
        self.P = 0
        self.N = 0
        self.W = 0
        self.action = -1 
        self.parent = None
        self.children = {}
        self.is_terminal = False
        self.winner = None

    @property
    def Q(self):
        if self.N:
            return self.W / self.N
        return 0
    
    @property
    def is_leaf(self):
        return not self.children

    def update(self):
        pass


class Root(Node):

    def __init__(self, whose_turn, board):
        super().__init__(whose_turn)
        self.board = board


def mcts_action_selection(board, whose_turn, network, total_sim=20):
    '''
    board: the current board state
    whose_turn: whose turn is it to take the next move
    network: a PolicyNetwork that outputs a distribution over actions and a value
    total_sim: how many total simulations do we want
    '''
    num_sim = 0
    root = Root(whose_turn=whose_turn, board=board)
    cur_node = root

    while(num_sim < total_sim):
        if cur_node.is_leaf:
            # There are two cases when a Node has no children
            # 1. It has not been expanded yet, but game isn't over.
            # 2. Game is at terminal state.
            if not cur_node.is_terminal:
                pass
            else:
                pass
        else:
            pass

