from camembert_finetune.env.imports import os, pdb, OrderedDict
from camembert_finetune.io_.dat import conllu_data
from camembert_finetune.trainer.tools.multi_task_tools import get_vocab_size_and_dictionary_per_task
from camembert_finetune.model.settings import TASKS_PARAMETER
from camembert_finetune.model.architecture.get_model import make_bert_multitask
from camembert_finetune.env.dir.tuned_model_dir import TUNED_MODEL_INVENTORY
from transformers.tokenization_camembert import CamembertTokenizer


def get_model_dirs(args):
    if args.checkpoint_dir is None:
        model_dir = TUNED_MODEL_INVENTORY[args.task]["dir"]
    else:
        print("LOADING: checkpoint directory passed manually")
        model_dir = args.checkpoint_dir
    assert os.path.isdir(model_dir), f"ERROR {model_dir} do not exist : it should include the checkpoint of {args.task}"

    if args.dict_dir is None:
        dict_path = model_dir + "/dictionaries"
    else:
        print("LOADING: dictionary directory passed manually")
        dict_path = args.dict_path
    assert os.path.isdir(dict_path), f"ERROR {dict_path} do not exist : it should include the dictionary for labels"

    if args.init_args_dir is None:
        args.init_args_dir = model_dir + "/args.json"
    else:
        print("LOADING: init_args_dir passed manually")
    assert os.path.isfile(args.init_args_dir), f"ERROR {args.init_args_dir} do not exist : it should be a json with the tune model arguments"

    return args, dict_path, model_dir


def load_tok_model_for_prediction(args):

    args.train_path = args.input_file
    dictionaries = {}
    # args.dev_path = args.test_paths
    if args.task == "ner":
        args.bert_model = "camembert-cased-oscar-wwm-107075step"# POS registered model
        hugging_face_name = "camembert-base"
    elif args.task == "pos":
        # temporarly  : the tuned models are from different pretrained camembert
        args.bert_model = "camembert-cased-1"
        hugging_face_name = "camembert/camembert-base-ccnet"
    elif args.task == "parsing":
        # temporarly  : the tuned models are from different pretrained camembert
        args.bert_model = "camembert-cased-oscar-wwm-107075step"  # POS registered model
        hugging_face_name = "camembert-base"
    else:
        raise(Exception("Not supported yet"))

    args.tasks = [[args.task if args.task != "ner" else "pos"]] # NER is identified as a pos
    assert args.task in TUNED_MODEL_INVENTORY.keys(), f"ERROR {args.task} should be in {TUNED_MODEL_INVENTORY.keys()}"

    # PARSING
    args, dict_path, model_dir = get_model_dirs(args)
    # args = args_preprocessing(args)

    encoder = 'RobertaModel' #BERT_MODEL_DIC[args.bert_model]["encoder"]
    vocab_size = 32005 #BERT_MODEL_DIC[args.bert_model]["vocab_size"]

    tokenizer = CamembertTokenizer

    args.output_attentions = False

    tokenizer = tokenizer.from_pretrained(hugging_face_name, do_lower_case=args.case == "lower")

    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=None,
                              dev_path=None,
                              test_path=None,
                              word_embed_dict={},
                              dry_run=False,
                              expand_vocab=False,
                              word_normalization=True,
                              force_new_dic=False,
                              tasks=args.tasks,
                              pos_specific_data_set=None,
                              # pos_specific_data_set=args.train_path[1] if len(args.tasks) > 1 and len(
                              #    args.train_path) > 1 and "pos" in args.tasks else None,
                              case=args.case,
                              # if not normalize pos or parsing in tasks we don't need dictionary
                              do_not_fill_dictionaries=len(set(["normalize", "pos", "parsing"]) & set(
                                  [task for tasks in args.tasks for task in tasks])) == 0,
                              add_start_char=None,
                              verbose=1)

    num_labels_per_task, task_to_label_dictionary = get_vocab_size_and_dictionary_per_task(
        [task for tasks in args.tasks for task in tasks],
        vocab_bert_wordpieces_len=vocab_size,
        pos_dictionary=pos_dictionary,
        type_dictionary=type_dictionary,
        task_parameters=TASKS_PARAMETER)

    model = make_bert_multitask(args=args, pretrained_model_dir=model_dir, init_args_dir=args.init_args_dir,
                                tasks=[task for tasks in args.tasks for task in tasks],
                                mask_id=tokenizer.mask_token_id, encoder=encoder,
                                num_labels_per_task=num_labels_per_task,
                                hugging_face_name=hugging_face_name,
                                model_dir=model_dir)
    dictionaries["word"] = word_dictionary
    dictionaries["word_norm"] = word_norm_dictionary
    dictionaries["char"] = char_dictionary
    dictionaries["pos"] = pos_dictionary
    dictionaries["xpos"] = xpos_dictionary
    dictionaries["type"] = type_dictionary

    return model, tokenizer, task_to_label_dictionary, dictionaries, vocab_size


def detokenized_src_label(source_preprocessed, predict_dic, label_ls, label_dic=None,
                          special_after_space_flag="▁",
                          input_key="wordpieces_inputs_words"):
    """
    Re-alignement from
    sub-word tokenized --> word
    all predictions    --> first token of each word prediction
    + removing first and last special characters
    :param source_preprocessed:
    :param predict_dic:
    :param label_dic:
    :return:
    """
    detokenized_source_preprocessed = OrderedDict([(input_key, [])])
    detokenized_label_batch = OrderedDict([(key, []) for key in predict_dic.keys()])
    gold_label_batch = OrderedDict([(key, []) for key in predict_dic.keys()])

    for batch_i, batch in enumerate(source_preprocessed[input_key]):
        # for src in batch[1:-1]:
        src = batch[1:-1]
        detokenized_src = []
        for ind, (label, pred_label) in enumerate(zip(label_ls, predict_dic)):
            # remove first and last specilal token

            prediction = predict_dic[pred_label][0][batch_i][1:-1]

            if label_dic is not None:
                gold = label_dic[pred_label.split("-")[0]][batch_i][1:-1]
                assert len(prediction) == len(gold)

            detokenized_label = []
            detokenized_gold = []

            try:
                assert len(prediction) == len(src), f"ERROR should have 1-1 alignement here " \
                f"for word level task prediction {len(prediction)} {len(src)}"

            except Exception as e:
                pdb.set_trace()
                print(Exception(e))

            if label_dic is None:
                for subword, label in zip(src, prediction):

                    if subword[0] != special_after_space_flag:
                        # we are in the middle of a token : so we ignore the label and we join strings
                        if ind == 0:
                            # ind 0 in case several label set --> we work on src for only 1 of them
                            # we build detokenized_src only for the first label type
                            if subword not in ["</s>", "<pad>"]:
                                # Handling special case where number have been splitted and failed to be reconstructed
                                try:
                                    fix = isinstance(eval("1" + subword), int)
                                except Exception as e:
                                    fix = False
                                # if its a number and label is -1 (pointed as non-first sub-token) then we need to fix
                                if fix and len(detokenized_src[-1]) == 0 and detokenized_label[-1] == -1:
                                    detokenized_label = detokenized_label[:-1]
                                    detokenized_src = detokenized_src[:-1]

                                detokenized_src[-1] += subword
                    else:
                        detokenized_label.append(label)
                        try:
                            fix = isinstance(eval("1" + subword[1:]), int)
                        except Exception as e:
                            fix = False
                        # we build detokenized_src only for the first label type
                        #print("Label append", detokenized_label)
                        if ind == 0:
                            if fix and detokenized_label[-1] == -1:
                                detokenized_label = detokenized_label[:-1]
                                detokenized_src[-1] += subword[1:]
                            else:
                                detokenized_src.append(subword[1:])
                detokenized_label_batch[pred_label].append(detokenized_label)
            else:
                for subword, label, gold in zip(src, prediction, gold):
                    if subword[0] != special_after_space_flag:
                        # we are in the middle of a token : so we ignore the label and we join strings
                        if ind == 0:
                            # we build detokenized_src only for the first label type
                            if subword not in ["</s>", "<pad>"]:
                                detokenized_src[-1] += subword
                    else:
                        detokenized_label.append(label)
                        detokenized_gold.append(gold)
                        # we build detokenized_src only for the first label type
                        if ind == 0:
                            # we remove the special character
                            detokenized_src.append(subword[1:])

                detokenized_label_batch[pred_label].append(detokenized_label)
                gold_label_batch[pred_label].append(detokenized_gold)

            if ind == 0:
                assert len(detokenized_src) == len(detokenized_label), "Should be aligned"
                detokenized_source_preprocessed[input_key].append(detokenized_src)
                if label_dic is not None:
                    assert len(detokenized_gold) == len(detokenized_label)

    def sanity_check_batch():
        batch_size = -1
        # checking that input and output have same batch size and all labels
        for key in predict_dic.keys():
            if batch_size != -1:
                assert len(detokenized_label_batch[key]) == batch_size
                if len(gold_label_batch[key]) > 0:
                    assert len(gold_label_batch[key]) == batch_size
            batch_size = len(detokenized_label_batch[key])
        assert len(detokenized_source_preprocessed[input_key]) == batch_size

    sanity_check_batch()

    return detokenized_source_preprocessed, detokenized_label_batch, gold_label_batch

