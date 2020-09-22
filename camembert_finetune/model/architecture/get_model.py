from camembert_finetune.env.imports import logging, tarfile, tempfile, torch, json, pdb, nn, OrderedDict, os

from camembert_finetune.io_.logger import printing
from camembert_finetune.io_.report.report_tools import get_init_args_dir
from camembert_finetune.transformers.transformers.modeling_multitask import BertMultiTask
from transformers.modeling_camembert import CamembertModel
from transformers.configuration_camembert import CamembertConfig



def make_bert_multitask(pretrained_model_dir, tasks, num_labels_per_task, init_args_dir, mask_id, encoder=None, args=None, model_dir=None, hugging_face_name=None):
    assert num_labels_per_task is not None and isinstance(num_labels_per_task, dict), \
        "ERROR : num_labels_per_task {} should be a dictionary".format(num_labels_per_task)
    assert isinstance(tasks, list) and len(tasks) >= 1, "ERROR tasks {} should be a list of len >=1".format(tasks)
    # we modify programmatically the config file base on argument passed to args

    if pretrained_model_dir is not None and init_args_dir is None:
        raise(Exception("Not supported yet"))
        # hugly but handling specific heritage of XLMModel (should be made better!)
        #multitask_wrapper = BertMultiTask#BertMultiTaskXLM if encoder == "XLMModel" else BertMultiTask
        #printing("WARNING : as encoder is {} using {} ", var=["XLMModel", multitask_wrapper], verbose=1, verbose_level=1)
        #model = multitask_wrapper.from_pretrained(pretrained_model_dir, tasks=tasks, mask_id=mask_id,
        #                                          num_labels_per_task=num_labels_per_task, mapping_keys_state_dic=DIR_2_STAT_MAPPING[pretrained_model_dir],
        #                                          encoder=eval(encoder), dropout_classifier=args.dropout_classifier,
        #                                          hidden_dropout_prob=args.hidden_dropout_prob, random_init=False)

    elif init_args_dir is not None:
        init_args_dir = get_init_args_dir(init_args_dir)
        args_checkpoint = json.load(open(init_args_dir, "r"))
        #assert "checkpoint_dir" in args_checkpoint, "ERROR checkpoint_dir not in {} ".format(args_checkpoint)

        #checkpoint_dir = args_checkpoint.get("checkpoint_dir")
        #if checkpoint_dir is None or not os.path.isfile(checkpoint_dir):
        assert model_dir is not None
        checkpoint_dir = model_dir+"/"+"checkpoint.pt"
        assert os.path.isfile(checkpoint_dir), f"ERROR checkpoint file was not found {checkpoint_dir} "
        # redefining model and reloading

        encoder = CamembertModel
        config = CamembertConfig.from_pretrained(hugging_face_name)

        model = BertMultiTask(config=config, tasks=[task for tasks in args_checkpoint["hyperparameters"]["tasks"] for task in tasks], num_labels_per_task=args_checkpoint["info_checkpoint"]["num_labels_per_task"],
                              encoder=encoder, mask_id=mask_id)
        printing("MODEL : loading model from checkpoint {}", var=[checkpoint_dir], verbose=1, verbose_level=1)
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        model.append_extra_heads_model(downstream_tasks=tasks, num_labels_dic_new=num_labels_per_task)
    else:
        raise(Exception("only one of pretrained_model_dir checkpoint_dir can be defined "))

    return model


def get_model_multi_task_bert(args, model_dir, mask_id, encoder, num_labels_per_task=None):
    # we flatten the tasks to make the model (we don't need to know if tasks are simulateneaous or not )
    model = make_bert_multitask(args=args, pretrained_model_dir=model_dir, init_args_dir=args.init_args_dir,
                                tasks=[task for tasks in args.tasks for task in tasks],
                                mask_id=mask_id,encoder=encoder,
                                num_labels_per_task=num_labels_per_task)
   
    return model
