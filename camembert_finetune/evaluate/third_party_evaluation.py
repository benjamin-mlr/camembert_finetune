from camembert_finetune.env.imports import os, OrderedDict


def evaluate_ner(dir_end_pred,
                 prediction_file, gold_file_name,
                 root=".", verbose=1,
             ):
    "cp from line 382 https://github.com/ufal/acl2019_nested_ner/blob/master/tagger.py"

    f1 = 0.0
    dataset_name = "dev"
    logdir = dir_end_pred

    dir_run_conll_eval = logdir
    assert os.path.isdir(dir_run_conll_eval), "ERROR {} does not exit".format(dir_run_conll_eval)
    if verbose > 2:
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


def evaluate_parsing(prediction_file, gold_file_name, dir_end_temp, task, verbose=1):
    assert task in ["pos", "parsing"]
    if task == "pos":
        result_to_get = ["UPOS"]
    else:
        result_to_get = ["UAS", "LAS"]
    final_score = OrderedDict()

    dir = os.path.dirname(os.path.abspath(__file__))
    script_eval_dir = os.path.join(dir, "conll18_ud_eval-modified.py")
    dir_end_temp = f"{dir_end_temp}/eval_temp_parse.txt"
    os.system(f"python {script_eval_dir} {gold_file_name} {prediction_file} -v > {dir_end_temp} ")
    # parsing report txt file to get the relevant score
    with open(dir_end_temp, "r") as result_file:
        for line in result_file:
            line = line.strip().replace(" ","").split("|")
            if line[0] in result_to_get:
                final_score[line[0]] = line[3]
    return final_score

