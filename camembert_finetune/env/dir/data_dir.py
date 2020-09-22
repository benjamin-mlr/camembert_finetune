

from camembert_finetune.env.imports import os, re



# NB : lots of the datasets directory are in project_variables
# this files aims to group all of those at some point


DATA_UD = os.environ.get("DATA_UD")
DATA_UD_25 = os.environ.get("DATA_UD_25")
DATA_WIKI_NER = os.environ.get("DATA_WIKI_NER")


DATASET_CODE_LS = ['af_afribooms']



def get_dir_data0(set, data_code, demo=False):

    assert set in ["train", "dev", "test"]
    assert data_code in DATASET_CODE_LS, "ERROR {}".format(data_code)
    demo_str = "-demo" if demo else ""

    file_dir = os.path.join(DATA_UD, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))

    assert os.path.isfile(file_dir), "{} not found".format(file_dir)

    return file_dir


def get_dir_data(set, data_code, demo=False):

    assert set in ["train", "dev", "test"], "{} - {}".format(set, data_code)
    assert data_code in DATASET_CODE_LS, "ERROR {}".format(data_code)
    demo_str = "-demo" if demo else ""
    # WE ASSSUME DEV AND TEST CANNOT FINISH by INTERGER_INTERGER IF THEY DO --> fall back to data_code origin
    if set in ["dev", "test"]:
        matching = re.match("(.*)_([0-9]+)_([0-9]+)$",data_code)
        if matching is not None:
            data_code = matching.group(1)
            print("WARNING : changed data code with {}".format(data_code))
        else:
            pass#print("DATA_CODE no int found  {} ".format(data_code))
    file_dir = os.path.join(DATA_UD, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))
    try:
        assert os.path.isfile(file_dir), "{} not found".format(file_dir)
    except:
        try:
            file_dir = os.path.join(DATA_UD_25, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))
            assert os.path.isfile(file_dir), "{} not found".format(file_dir)
            print("WARNING : UD 25 USED ")
        except Exception as e:
            print("--> data ", e)
            demo_str = ""
            file_dir = os.path.join(DATA_WIKI_NER, "{}-wikiner-{}{}.conll".format(data_code, set, demo_str))
            assert os.path.isfile(file_dir), "{} not found".format(file_dir)
            print("WARNING : WIKI NER USED")
    return file_dir



def get_code_data(dir):
    matching = re.match(".*\/([^\/]+).*.conllu", dir)
    if matching is not None:
        return matching.group(1)
    return "training_set-not-found"

# 1 define list of dataset code and code 2 dir dictionary
# 2 : from grid_run : call dictionary and iterate on data set code

# DATA DICTIONARY

