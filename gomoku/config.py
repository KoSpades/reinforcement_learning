from pathlib import Path


BOARD_SIZE = 9
TRAIN_ITER = 1000
UI_ITER = 10000
TRAINING_ALGO = "actor_critic"

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
REINFORCE_MODELS_DIR = MODELS_DIR / "reinforce"
ACTOR_CRITIC_MODELS_DIR = MODELS_DIR / "actor_critic"
FROZEN_DIR = MODELS_DIR / "frozen"
MODEL_DIRS_BY_ALGO = {
    "reinforce": REINFORCE_MODELS_DIR,
    "actor_critic": ACTOR_CRITIC_MODELS_DIR,
}
CURRENT_MODELS_DIR = MODEL_DIRS_BY_ALGO[TRAINING_ALGO]
