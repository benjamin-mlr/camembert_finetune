import os

TUNED_MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "checkpoints")

#TUNED_MODEL_DIR = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo"

assert os.path.isdir(TUNED_MODEL_DIR), f"ERROR : {TUNED_MODEL_DIR} does not exist : mkdir and upload tuned models to it " \
        f"OR set it to the directory of your wish"


TUNED_MODEL_INVENTORY = {
    "pos":
        {"dir": f"{TUNED_MODEL_DIR}/pos"},
    "parsing":
        {"dir": f"{TUNED_MODEL_DIR}/parsing"},
    "ner":
        {"dir": f"{TUNED_MODEL_DIR}/ner"}

                         }
