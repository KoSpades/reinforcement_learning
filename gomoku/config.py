from pathlib import Path


BOARD_SIZE = 9
TRAIN_ITER = 10000
UI_ITER = 10000
TRAINING_ALGO = "actor_critic"

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
