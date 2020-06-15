

from camembert.downstream.finetune.env.imports import pdb, torch
from camembert.downstream.finetune.io_.logger import printing
import camembert.downstream.finetune.model.optimization.deep_learning_toolbox as dptx
from camembert.downstream.finetune.model.settings import AVAILABLE_BERT_FINE_TUNING_STRATEGY
#from training.bert_normalize.optimizer import WarmupLinearSchedule
from camembert.downstream.finetune.transformers.transformers.optimization import WarmupLinearSchedule


def apply_fine_tuning_strategy(fine_tuning_strategy, model, epoch, lr_init,optimizer_name,
                               betas=None, weight_decay=None, t_total =None, verbose=1):
    """
    get optimizers based on fine tuning strategies that might involve having several optimizers, and freezing some layers
    """

    assert fine_tuning_strategy in AVAILABLE_BERT_FINE_TUNING_STRATEGY, "{} not in {}".format(fine_tuning_strategy, AVAILABLE_BERT_FINE_TUNING_STRATEGY)
    scheduler = None
    if fine_tuning_strategy in ["standart", "bert_out_first", "only_first_and_last", "freeze"]:
        assert isinstance(lr_init, float), "{} lr : type {}".format(lr_init, type(lr_init))
        optimizer = [dptx.get_optimizer(model.parameters(), lr=lr_init, betas=betas, weight_decay=weight_decay, optimizer=optimizer_name)]

        if optimizer_name == "AdamW":
            assert t_total is not None
            assert len(optimizer) == 1, "ERROR scheduler only supported when 1 optimizer "
            printing("OPTIMIZING warmup_steps:{} t_total:{}", var=[t_total / 10, t_total], verbose=verbose,
                     verbose_level=1)
            scheduler = WarmupLinearSchedule(optimizer[0], warmup_steps=t_total / 10, t_total=t_total)  # PyTorch scheduler

        printing("TRAINING : fine tuning strategy {} : learning rate constant {} betas {}", var=[fine_tuning_strategy, lr_init, betas],
                 verbose_level=1, verbose=verbose)

    elif fine_tuning_strategy == "flexible_lr":
        assert isinstance(lr_init, dict), "lr_init should be dict in {}".format(fine_tuning_strategy)
        # sanity check

        assert optimizer_name in ["adam"], "ERROR only adam supporte in flexible_lr"

        optimizer = []
        n_all_layers = len([a for a, _ in model.named_parameters()])
        n_optim_layer = 0
        for pref, lr in lr_init.items():
            param_group = [param for name, param in model.named_parameters() if name.startswith(pref)]
            n_optim_layer += len(param_group)
            optimizer.append(dptx.get_optimizer(param_group, lr=lr, betas=betas, optimizer=optimizer_name))
        assert n_all_layers == n_optim_layer, \
            "ERROR : You are missing some layers in the optimization n_all {} n_optim {} ".format(n_all_layers,
                                                                                                  n_optim_layer)

        printing("TRAINING : fine tuning strategy {} : learning rate constant : {} betas {}", var=[fine_tuning_strategy,
                                                                                                   lr_init, betas],
                 verbose_level=1, verbose=verbose)

    if fine_tuning_strategy in ["bert_out_first", "freeze"]:
        info_add = ""
        if (epoch <= 1 and fine_tuning_strategy == "bert_out_first") or fine_tuning_strategy == "freeze":
            info_add = "not"
            freeze_layer_prefix_ls = "encoder"
            model = dptx.freeze_param(model, freeze_layer_prefix_ls, verbose=verbose)

        printing("TRAINING : fine tuning strategy {} : {} freezing bert for epoch {}"\
                 .format(fine_tuning_strategy, info_add, epoch), verbose_level=1, verbose=verbose)
    elif fine_tuning_strategy == "only_first_and_last":
        #optimizer = [torch.optim.Adam(model.parameters(), lr=lr_init, betas=betas, eps=1e-9)]
        model = dptx.freeze_param(model, freeze_layer_prefix_ls=None,
                                  not_freeze_layer_prefix_ls=["embeddings", "classifier"],
                                  verbose=verbose)

    return model, optimizer, scheduler
