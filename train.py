
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

from camembert.downstream.finetune.args.args_parse import args_train, args_preprocessing
from camembert.downstream.finetune.trainer.train_predict_runner import train_predict_eval

if __name__ == "__main__":

    args = args_train()

    args = args_preprocessing(args)

    train_predict_eval(args)
