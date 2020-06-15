from camembert_finetune.env.imports import argparse, OrderedDict, re
from camembert_finetune.args.args_settings import DIC_ARGS, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE, AVAILALE_PENALIZATION_MODE
from camembert_finetune.model.settings import TASKS_PARAMETER
from camembert_finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from camembert_finetune.io_.logger import printing


def get_config_param_to_modify(args):
    """ for now only internal bert dropout can be modifed a such"""
    config_to_update = OrderedDict()
    if args.dropout_bert is not None:
        assert args.dropout_bert >= 0, "ERROR {}".format(args.dropout_bert)
        config_to_update["attention_probs_dropout_prob"] = args.dropout_bert
        config_to_update["hidden_dropout_prob"] = args.dropout_bert
    if args.dropout_classifier is not None:
        assert args.dropout_classifier >= 0
        config_to_update["dropout_classifier"] = args.dropout_classifier
    if args.graph_head_hidden_size_mlp_rel is not None:
        config_to_update["graph_head_hidden_size_mlp_rel"] = args.graph_head_hidden_size_mlp_rel
    if args.graph_head_hidden_size_mlp_rel is not None:
        config_to_update["graph_head_hidden_size_mlp_arc"] = args.graph_head_hidden_size_mlp_arc

    return config_to_update


def parse_argument_dictionary(argument_as_string, logits_label=None, hyperparameter="multi_task_loss_ponderation", verbose=1):
    """
    All arguments that are meant to be defined as dictionaries are passed to the Argument Parser as string:
    following  template :  i.e 'key1=value1,key2=value,'  (matched with "{}=([^=]*),".format(sub) )
    ALl the dictionary arguments are listed in DIC_ARGS
    """
    assert hyperparameter in DIC_ARGS, "ERROR only supported"
    if argument_as_string in MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE:
        return argument_as_string
    else:
        dic = OrderedDict()
        if hyperparameter == "multi_task_loss_ponderation":
            assert logits_label is not None
            for task in logits_label:
                # useless (I think)
                if task == "parsing":
                    for sub in ["parsing-heads", "parsing-types"]:
                        pattern = "{}=([^=]*),".format(sub)
                        match = re.search(pattern, argument_as_string)
                        assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(
                            pattern, task, argument_as_string)
                        dic[sub] = eval(match.group(1))
                # useless (I thinh)
                elif task == "normalize":
                    for sub in ["normalize", "append_masks"]:
                        pattern = "{}=([^=]*),".format(sub)
                        match = re.search(pattern, argument_as_string)
                        if sub == "normalize":
                            assert match is not None, "ERROR : pattern {} not found for task {} " \
                                                      "in argument_as_string {}  ".format( pattern, task, argument_as_string)
                            dic[sub] = eval(match.group(1))
                        else:
                            if match is not None:
                                dic[sub] = eval(match.group(1))
                # all cases should be in this one
                if task != "all" and task != "parsing":

                    pattern = "{}=([^=]*),".format(task)
                    match = re.search(pattern, argument_as_string)
                    assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(pattern, task, argument_as_string)
                    dic[task] = eval(match.group(1))

            printing("SANITY CHECK : multi_task_loss_ponderation {} ", var=[argument_as_string], verbose_level=3, verbose=verbose)

        elif hyperparameter in ["lr", "norm_order_per_layer", "ponderation_per_layer"]:
            # to handle several optimizers
            try:
                assert isinstance(eval(argument_as_string), float)
                return eval(argument_as_string)
            except:
                argument_as_string = argument_as_string.split(",")
                for arg in argument_as_string[:-1]:
                    # DIFFERENCE WITH ABOVE IS THE COMMA
                    pattern = "([^=]*)=([^=]*)"
                    match = re.search(pattern, arg)
                    assert match is not None, "ERROR : pattern {} not found in argument_as_string {}  ".format(pattern,arg)
                    if hyperparameter in ["lr"]:
                        dic[match.group(1)] = float(match.group(2))
                    elif hyperparameter in ["norm_order_per_layer"]:
                        if match.group(2) != "fro":
                            dic[match.group(1)] = float(match.group(2))
                        else:
                            dic[match.group(1)] = match.group(2)
                    elif hyperparameter in ["ponderation_per_layer"]:
                        dic[match.group(1)] = float(match.group(2))

        return dic


def args_preprocessing(args, verbose=1):
    """
    sanity checking , changing types of arguments and parsing arguments
    """

    if isinstance(args.schedule_lr, str) and args.schedule_lr=="None":
        args.schedule_lr = eval(args.schedule_lr)

    if args.batch_size != "flexible":
        args.batch_size = int(args.batch_size)

    if args.low_memory_foot_print_batch_mode is not None and args.low_memory_foot_print_batch_mode != "flexible_forward_batch_size":
        args.low_memory_foot_print_batch_mode = int(args.low_memory_foot_print_batch_mode)
    low_memory_foot_print_batch_mode_available = [0, 1, "flexible_forward_batch_size"]

    assert args.low_memory_foot_print_batch_mode is None or args.low_memory_foot_print_batch_mode in low_memory_foot_print_batch_mode_available, "ERROR args.low_memory_foot_print_batch_mode {} should be in {}".format(args.low_memory_foot_print_batch_mode, low_memory_foot_print_batch_mode_available)

    if args.low_memory_foot_print_batch_mode:
        args.batch_update_train = args.batch_size
        args.batch_size = "flexible" if args.low_memory_foot_print_batch_mode == "flexible_forward_batch_size" else 2
        printing("INFO : args.low_memory_foot_print_batch_mode {} "
                 "so setting batch_size to {} and args.batch_update_train {}",
                 var=[args.low_memory_foot_print_batch_mode, args.batch_size, args.batch_update_train],
                 verbose=verbose, verbose_level=1)

        if args.low_memory_foot_print_batch_mode != "flexible_forward_batch_size":
            assert isinstance(args.batch_update_train // args.batch_size,
                              int) and args.batch_update_train // args.batch_size > 0, "ERROR batch_size {} should be a multiple of 2 ".format(args.batch_update_train)
        printing("INFO iterator : updating with {} equivalent batch size : forward pass is {} batch size",
                 var=[args.batch_update_train, args.batch_size], verbose=verbose, verbose_level=1)
    else:
        args.batch_update_train = args.batch_size
    params = vars(args)
    args.lr = parse_argument_dictionary(params["lr"], hyperparameter="lr")

    if args.test_paths is not None:
        args.test_paths = [test_path_task.split(",") for test_path_task in args.test_paths]

    if args.dev_path is not None:
        args.dev_path = [dev_path_task.split(",") for dev_path_task in args.dev_path]

    if args.ponderation_per_layer is not None:
        args.ponderation_per_layer = parse_argument_dictionary(params["ponderation_per_layer"],
                                                               hyperparameter="ponderation_per_layer")
    if args.norm_order_per_layer is not None:
        args.norm_order_per_layer = parse_argument_dictionary(params["norm_order_per_layer"],
                                                              hyperparameter="norm_order_per_layer")

    args.tasks = [task_simul.split(",") for task_simul in args.tasks]

    if args.test_paths is not None:
        assert isinstance(args.test_paths, list) and isinstance(args.test_paths[0], list), "ERROR args.test_paths should be a list"
    # 1 simultaneous set of tasks per training dataset
    assert len(args.tasks) == len(args.train_path), "ERROR args.tasks is {} but train paths are {}".format(args.tasks, args.train_path)

    assert args.penalization_mode in AVAILALE_PENALIZATION_MODE, "ERROR args.penalization_mode {} should be in {}".format(args.penalization_mode, AVAILALE_PENALIZATION_MODE)

    if args.multi_task_loss_ponderation is not None:
        argument_as_string = args.multi_task_loss_ponderation
        assert args.tasks is not None
        tasks = [task for tasks in args.tasks for task in tasks]
        # should add test on task X label calling task setting
        for task in tasks:
            if task != "all":
                for label in TASKS_PARAMETER[task]["label"]:
                    pattern = "{}-{}=([^=]*),".format(task, label)
                    match = re.search(pattern, argument_as_string)
                    assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(pattern, task, argument_as_string)

    if args.bert_model is not None:
        assert args.bert_model in BERT_MODEL_DIC, "ERROR args.bert_model {} should be in {}".format(args.bert_model, BERT_MODEL_DIC.keys())

    return args


def args_train(mode="command_line", training=True):
    """

    :param mode:
    :return:
    """

    assert mode in ["command_line", "script"], "mode should be in '[command_line, script]"

    parser = argparse.ArgumentParser()

    # training opti
    parser.add_argument("--batch_size", default=1, required=training)
    parser.add_argument("--epochs", default=None, type=int,required=training)
    parser.add_argument("--lr", default="0.1",required=training)

    parser.add_argument("--optimizer", default="adam", help="display a square of a given number ")

    # id and reporting
    parser.add_argument("--model_id_pref", required=training)
    parser.add_argument("--overall_label", default="D")
    parser.add_argument("--overall_report_dir", required=training, help="display a square of a given number")
    # logging and debugging
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--verbose", default=1, type=int)

    parser.add_argument("--gpu", default=None, type=str)
    # data
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--train_path", required=training, nargs='+', help='<Required> Set flag')
    parser.add_argument("--dev_path", required=training, nargs='+', help='<Required> Set flag')
    parser.add_argument('--test_paths', nargs='+', help='<Required> Set flag', default=None, required=True)

    parser.add_argument('--end_predictions', default=None, required=not training)
    parser.add_argument('--multitask', type=int, default=1)

    parser.add_argument('--tasks', nargs='+', help='<Required> Set flag')

    parser.add_argument("--seed", default=42, type=int, help="seed of pytorch and numpy ")

    parser.add_argument("--initialize_bpe_layer", default=1, type=int )
    parser.add_argument("--bert_model", required=training, type=str )
    parser.add_argument("--freeze_parameters", default=0, type=int )
    parser.add_argument('--freeze_layer_prefix_ls', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--dropout_classifier', type=float, default=None)
    parser.add_argument('--fine_tuning_strategy', type=str, default="standart")
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--heuristic_ls', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--gold_error_detection', type=int, default=0)
    parser.add_argument('--dropout_input_bpe', type=float, default=0.)
    parser.add_argument('--dropout_bert', type=float, default=None)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)

    parser.add_argument('--masking_strategy', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--portion_mask', type=float, default=None)
    parser.add_argument('--init_args_dir', type=str, default=None)
    
    parser.add_argument('--norm_2_noise_training', type=float, default=None)
    parser.add_argument('--aggregating_bert_layer_mode', type=str, default=None)

    parser.add_argument('--bert_module', type=str, default=None)
    parser.add_argument('--layer_wise_attention', type=int, default=0)

    parser.add_argument('--tokenize_and_bpe', type=int, default=0)
    parser.add_argument('--append_n_mask', type=int, default=0)

    parser.add_argument('--schedule_lr', type=str, default=None)
    parser.add_argument('--n_steps_warmup', type=int, default=0)

    parser.add_argument('--multi_task_loss_ponderation', type=str, default="pos-pos=1.0,parsing-types=1,parsing-heads=1,mlm-wordpieces_inputs_words=1.0,")

    parser.add_argument('--memory_efficient_iterator', type=int, default=0)

    parser.add_argument("--n_iter_max_train", default=1000000, type=int)
    parser.add_argument("--saving_every_n_epoch", default=100, type=int)
    parser.add_argument("--name_inflation", default=0, type=int)
    parser.add_argument("--demo_run", default=0, type=int, help="means running for 5 iteration max ")

    parser.add_argument("--case", default=None, type=str, help="means running for 5 iteration max ")
    parser.add_argument("--low", default=None, type=str, help="means running for 5 iteration max ")
    parser.add_argument("--low_memory_foot_print_batch_mode", default=0, help="means running for 5 iteration max ")

    parser.add_argument("--graph_head_hidden_size_mlp_rel", default=None, type=int)
    parser.add_argument("--graph_head_hidden_size_mlp_arc", default=None, type=int)

    parser.add_argument("--penalize", default=1, type=int)

    parser.add_argument("--norm_order_per_layer", default=None)
    parser.add_argument("--ponderation_per_layer", default=None)

    parser.add_argument("--random_init", default=0, type=int)

    parser.add_argument("--penalization_mode", default="layer_wise")

    args = parser.parse_args()

    return args


def args_check(args):
    """
    sanity checking and setting default behavior of arguments
    :param args:
    :return:
    """
    if args.mode is None:
        if args.input_file is not None or args.output_file is not None:
            args.mode = "conll"
        else:
            args.mode = "interactive"

    if args.mode == "conll":

        assert args.input_file is not None, \
            "ERROR : --input_file required in mode 'conll', directory of the input conll file"
        assert args.output_file is not None, \
            "ERROR : --output_file required in mode 'conll', directory to write prediction " \
            "(WARNING : if file exists it will overwrite it with predictions)"

    elif args.mode == "interactive":
        if args.input_file:
            print("WARNING: --input_file will be ignore because 'interactive' mode")
        if args.output_file:
            print("WARNING: --output_file will be ignore because 'interactive' mode")

    else:
        raise(Exception(f"{args.mode} not supported"))


def args_predict(mode="interactive"):
    """

    :param mode:
    :return:
    """

    assert mode in ["interactive", "script"], "mode should be in '[interactive, script]"

    parser = argparse.ArgumentParser()

    # training opti
    parser.add_argument("--batch_size", default=6, required=False, type=int)
    parser.add_argument("--mode", default=None, required=False)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument('--input_file', nargs='+', help='<Required> Set flag', default=None, required=False)
    parser.add_argument('--output_file', help="Help : should define output_file directory for writing predictions", type=str, default=None, required=False)  # mode=="script")
    parser.add_argument('--gold_file', help="Help : should define gold_file directory for evaluation",
                        type=str, default=None, required=False)  # mode=="script")
    parser.add_argument('--score_details',
                        type=int, default=0, required=False)  # mode=="script")

    parser.add_argument("--gpu", default=None, type=str)

    parser.add_argument('--end_predictions', default=None, required=False)
    parser.add_argument('--task', default="pos", required=False)
    parser.add_argument('--tasks', default=None, required=False)
    parser.add_argument("--seed", default=42, type=int, help="seed of pytorch and numpy ")
    parser.add_argument("--case", default=None, type=str, help="means running for 5 iteration max ")

    # should remain None
    parser.add_argument('--dropout_bert', type=float, default=None)
    parser.add_argument('--dropout_classifier', type=float, default=None)
    parser.add_argument("--graph_head_hidden_size_mlp_rel", default=None, type=int)
    parser.add_argument("--graph_head_hidden_size_mlp_arc", default=None, type=int)

    # NEW argument for loading
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--init_args_dir", default=None, type=str)
    parser.add_argument("--dict_dir", default=None, type=str)

    # NEW UX argument
    parser.add_argument("--output_tokenized", default=0, type=int)
    #parser.add_argument("--epochs", default=None, type=int, required=training)
    #parser.add_argument("--lr", default="0.1", required=training)

    #parser.add_argument("--optimizer", default="adam", help="display a square of a given number ")

    # id and reporting
    #parser.add_argument("--model_id_pref", required=training)
    #parser.add_argument("--overall_label", default="D")
    #parser.add_argument("--overall_report_dir", required=training, help="display a square of a given number")
    # logging and debugging
    #parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--warmup", action="store_true")



    # data

    #parser.add_argument("--train", default=1, type=int)
    #parser.add_argument("--train_path", required=training, nargs='+', help='<Required> Set flag')
    #parser.add_argument("--dev_path", required=training, nargs='+', help='<Required> Set flag')

    #parser.add_argument('--multitask', type=int, default=1)

    #parser.add_argument('--tasks', nargs='+', help='<Required> Set flag')


    #parser.add_argument("--initialize_bpe_layer", default=1, type=int)
    #parser.add_argument("--bert_model", required=training, type=str)
    #parser.add_argument("--freeze_parameters", default=0, type=int)
    #parser.add_argument('--freeze_layer_prefix_ls', nargs='+', help='<Required> Set flag', default=None)

    #parser.add_argument('--fine_tuning_strategy', type=str, default="standart")
    #parser.add_argument('--weight_decay', type=float, default=0.0)

    #parser.add_argument('--heuristic_ls', nargs='+', help='<Required> Set flag', default=None)
    #parser.add_argument('--gold_error_detection', type=int, default=0)
    #parser.add_argument('--dropout_input_bpe', type=float, default=0.)

    #parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)

    #parser.add_argument('--masking_strategy', nargs='+', help='<Required> Set flag', default=None)
    #parser.add_argument('--portion_mask', type=float, default=None)
    #parser.add_argument('--init_args_dir', type=str, default=None)

    #parser.add_argument('--norm_2_noise_training', type=float, default=None)
    #parser.add_argument('--aggregating_bert_layer_mode', type=str, default=None)

    #parser.add_argument('--bert_module', type=str, default=None)
    #parser.add_argument('--layer_wise_attention', type=int, default=0)

    #parser.add_argument('--tokenize_and_bpe', type=int, default=0)
    #parser.add_argument('--append_n_mask', type=int, default=0)

    #parser.add_argument('--schedule_lr', type=str, default=None)
    #parser.add_argument('--n_steps_warmup', type=int, default=0)

    #parser.add_argument('--multi_task_loss_ponderation', type=str,
    #                     default="pos-pos=1.0,parsing-types=1,parsing-heads=1,mlm-wordpieces_inputs_words=1.0,")

    #parser.add_argument('--memory_efficient_iterator', type=int, default=0)

    #parser.add_argument("--n_iter_max_train", default=1000000, type=int)
    #parser.add_argument("--saving_every_n_epoch", default=100, type=int)
    #parser.add_argument("--name_inflation", default=0, type=int)
    #parser.add_argument("--demo_run", default=0, type=int, help="means running for 5 iteration max ")


    #parser.add_argument("--low", default=None, type=str, help="means running for 5 iteration max ")
    #parser.add_argument("--low_memory_foot_print_batch_mode", default=0, help="means running for 5 iteration max ")



    #parser.add_argument("--penalize", default=1, type=int)

    #parser.add_argument("--norm_order_per_layer", default=None)
    #parser.add_argument("--ponderation_per_layer", default=None)

    #parser.add_argument("--random_init", default=0, type=int)

    #parser.add_argument("--penalization_mode", default="layer_wise")

    args = parser.parse_args()

    return args
