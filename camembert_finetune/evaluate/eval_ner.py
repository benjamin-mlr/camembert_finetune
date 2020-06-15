from camembert_finetune.env.imports import os
from camembert_finetune.env.dir.project_directories import PROJECT_PATH


def evaluate(dir_end_pred,
             prediction_file, gold_file_name,
             root=".",verbose=1,
             ):
    "cp from line 382 https://github.com/ufal/acl2019_nested_ner/blob/master/tagger.py"

    f1 = 0.0
    dataset_name = "dev"
    logdir = dir_end_pred

    dir_run_conll_eval = logdir
    assert os.path.isdir(dir_run_conll_eval), "ERROR {} does not exit".format(dir_run_conll_eval)
    if verbose>2:
        print(f"cd {logdir} && ../run_conlleval.sh {dataset_name} {gold_file_name} {prediction_file}")
    os.system(f"cd {logdir} && ../run_conlleval.sh  {dataset_name} {gold_file_name} {prediction_file}")#.format(logdir, dir_run_conll_eval, dataset_name, gold_file_name, prediction_file))
    with open("{}/{}.eval".format(logdir, dataset_name), "r", encoding="utf-8") as result_file:
        for line in result_file:
            if verbose > 1:
                print(line.strip())
            line = line.strip("\n")
            if line.startswith("accuracy:"):
                f1 = float(line.split()[-1])

    return f1


