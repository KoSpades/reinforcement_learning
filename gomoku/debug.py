# Most of the code in this file is LLM generated. I just gave it instructions for what I want :)

import torch

from config import BOARD_SIZE, ACTOR_CRITIC_MODELS_DIR
from model import PolicyNetwork
from utils import step


def get_debug_board_specs():
    return [
        ("empty", []),
        ("short_center_probe", [40, 39, 41]),
        ("short_diagonal_probe", [31, 40, 49]),
        ("short_side_extension", [30, 40, 31]),
        ("medium_center_fight", [40, 39, 41, 31, 49, 32, 50, 48]),
        ("medium_diagonal_shape", [31, 40, 32, 41, 22, 49, 30, 50]),
        ("medium_side_attack", [29, 38, 30, 39, 31, 48, 32, 49]),
        (
            "long_center_battle",
            [40, 39, 41, 31, 49, 32, 50, 48, 30, 58, 42, 57, 33, 47, 51, 56, 43, 29, 60, 38],
        ),
        (
            "long_diagonal_battle",
            [31, 40, 32, 41, 22, 49, 30, 50, 21, 58, 39, 59, 48, 23, 57, 33, 66, 42, 24, 60],
        ),
        (
            "long_edge_transition",
            [27, 36, 28, 37, 29, 38, 30, 39, 31, 48, 32, 49, 40, 41, 57, 42, 58, 50, 66, 51],
        ),
        (
            "random_5",
            [4, 67, 29, 73, 41],
        ),
        (
            "random_10",
            [3, 52, 77, 18, 46, 9, 64, 25, 71, 34],
        ),
    ]


def make_board_from_actions(actions):
    state = torch.zeros((2, BOARD_SIZE, BOARD_SIZE))
    turn = 0
    for action in actions:
        state = step(state, action, turn)
        turn = 1 - turn
    return state


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
    linear = policy.value_head[1]
    tanh = policy.value_head[2]

    print("------ Value Outputs Across Boards ------")
    with torch.no_grad():
        for name, actions in get_debug_board_specs():
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
    boards = [make_board_from_actions(actions) for _, actions in get_debug_board_specs()]
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


def inspect_policy_move_diversity(policy):
    print("------ Policy Top-5 Moves Across Boards ------")
    with torch.no_grad():
        for name, actions in get_debug_board_specs():
            board = make_board_from_actions(actions)
            action_logits, _ = policy(board.unsqueeze(0))
            action_logits = action_logits.squeeze(0)
            occupied_spaces = board.sum(dim=0)
            legal_mask = (occupied_spaces == 0).flatten()
            masked_logits = action_logits.masked_fill(~legal_mask, float("-inf"))
            action_probs = torch.softmax(masked_logits, dim=-1)

            top_probs, top_actions = torch.topk(action_probs, k=min(5, int(legal_mask.sum().item())))
            formatted = []
            for action, prob in zip(top_actions.tolist(), top_probs.tolist()):
                if prob >= 0.01:
                    prob_pct = round(prob * 100)
                    formatted.append(
                        f"({action // BOARD_SIZE}, {action % BOARD_SIZE})={prob_pct}"
                    )
            if formatted:
                print(f"{name}: " + ", ".join(formatted))
            else:
                print(f"{name}: no legal moves above 1%")


def main():
    model_path = ACTOR_CRITIC_MODELS_DIR / "final_policy_1000.pt"
    policy = load_policy(model_path)
    inspect_value_head_linear(policy)
    inspect_value_outputs_across_boards(policy)
    inspect_policy_move_diversity(policy)
    inspect_dead_backbone_neurons(policy)


if __name__ == "__main__":
    main()
