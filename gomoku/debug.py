import torch

from config import BOARD_SIZE, ACTOR_CRITIC_MODELS_DIR
from model import PolicyNetwork
from utils import step


def load_policy(model_path):
    policy = PolicyNetwork().to("cpu")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def inspect_value_head_linear(policy):
    linear = policy.value_head[1]
    weights = linear.weight.detach().flatten()
    bias = linear.bias.detach().flatten()
    print("------ Value Head Linear Layer ------")
    print(f"weight_shape={tuple(linear.weight.shape)}")
    print(f"bias_shape={tuple(linear.bias.shape)}")
    print(
        "weights: "
        f"min={weights.min().item():.6f}, "
        f"max={weights.max().item():.6f}, "
        f"mean={weights.mean().item():.6f}, "
        f"std={weights.std().item():.6f}"
    )
    print(f"bias={bias.tolist()}")


def inspect_value_outputs_across_boards(policy):
    def make_board_from_actions(actions):
        state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE))
        turn = 0
        for action in actions:
            state = step(state, action, turn)
            turn = 1 - turn
        return state

    board_specs = [
        ("empty", []),
        ("center_opening", [40]),
        ("three_corners", [0, BOARD_SIZE**2 - 1, 40]),
        ("small_diagonal", [0, 1, 10]),
        ("main_diagonal", [0, 1, 10, 11, 20, 21]),
        ("anti_diagonal", [8, 7, 16, 15, 24, 23]),
        ("top_row_cluster", [0, 9, 1, 10, 2, 11, 3, 12, 4, 13]),
        ("cross_center", [40, 39, 31, 41, 49, 30, 50, 32, 48, 58]),
        (
            "spread_edges_15",
            [0, 8, 72, 80, 36, 44, 4, 76, 27, 53, 18, 62, 9, 71, 40],
        ),
        (
            "wide_mix_30",
            [
                0, 1, 8, 9, 16, 17, 24, 25, 32, 33,
                40, 41, 48, 49, 56, 57, 64, 65, 72, 73,
                2, 10, 18, 26, 34, 42, 50, 58, 66, 74,
            ],
        ),
    ]

    linear = policy.value_head[1]
    tanh = policy.value_head[2]

    print("------ Value Outputs Across Boards ------")
    with torch.no_grad():
        for name, actions in board_specs:
            board = make_board_from_actions(actions)
            features = policy.backbone(board.unsqueeze(0))
            flat_features = torch.flatten(features, 1)
            pre_tanh = linear(flat_features).item()
            post_tanh = tanh(torch.tensor([[pre_tanh]])).item()
            print(
                f"{name}: moves={len(actions)}, "
                f"pre_tanh={pre_tanh:.6f}, post_tanh={post_tanh:.6f}"
            )


def inspect_dead_backbone_neurons(policy):
    def make_board_from_actions(actions):
        state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE))
        turn = 0
        for action in actions:
            state = step(state, action, turn)
            turn = 1 - turn
        return state

    board_specs = [
        ("empty", []),
        ("center_opening", [40]),
        ("three_corners", [0, BOARD_SIZE**2 - 1, 40]),
        ("small_diagonal", [0, 1, 10]),
        ("main_diagonal", [0, 1, 10, 11, 20, 21]),
        ("anti_diagonal", [8, 7, 16, 15, 24, 23]),
        ("top_row_cluster", [0, 9, 1, 10, 2, 11, 3, 12, 4, 13]),
        ("cross_center", [40, 39, 31, 41, 49, 30, 50, 32, 48, 58]),
        ("spread_edges_15", [0, 8, 72, 80, 36, 44, 4, 76, 27, 53, 18, 62, 9, 71, 40]),
        (
            "wide_mix_30",
            [
                0, 1, 8, 9, 16, 17, 24, 25, 32, 33,
                40, 41, 48, 49, 56, 57, 64, 65, 72, 73,
                2, 10, 18, 26, 34, 42, 50, 58, 66, 74,
            ],
        ),
    ]

    boards = [make_board_from_actions(actions) for _, actions in board_specs]
    current = torch.stack(boards)
    relu_idx = 0

    print("------ Dead ReLU Check ------")
    with torch.no_grad():
        for layer in policy.backbone:
            current = layer(current)
            if isinstance(layer, torch.nn.ReLU):
                relu_idx += 1
                always_zero = (current == 0).all(dim=0)
                dead_positions = int(always_zero.sum().item())
                total_positions = always_zero.numel()
                dead_ratio = dead_positions / total_positions

                per_channel_dead = always_zero.view(always_zero.shape[0], -1).all(dim=1)
                dead_channels = int(per_channel_dead.sum().item())
                total_channels = int(per_channel_dead.numel())

                print(
                    f"ReLU {relu_idx}: "
                    f"dead_positions={dead_positions}/{total_positions} ({dead_ratio:.2%}), "
                    f"dead_channels={dead_channels}/{total_channels}"
                )


def main():
    model_path = ACTOR_CRITIC_MODELS_DIR / "final_policy_1000.pt"
    policy = load_policy(model_path)
    inspect_value_head_linear(policy)
    inspect_value_outputs_across_boards(policy)
    inspect_dead_backbone_neurons(policy)


if __name__ == "__main__":
    main()
