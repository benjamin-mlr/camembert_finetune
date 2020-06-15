

from camembert.downstream.finetune.env.imports import OrderedDict, time, pdb, os, re,torch
from camembert.downstream.finetune.io_.logger import printing
from camembert.downstream.finetune.evaluate.score.report_template import report_template
from camembert.downstream.finetune.trainer.tools.eval_ner import evaluate

from camembert.downstream.finetune.env.gpu_tools.gpu_info import printout_allocated_gpu_memory
from camembert.downstream.finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from camembert.downstream.finetune.io_.bert_iterators_tools.get_bpe_labels import get_label_per_bpe
from camembert.downstream.finetune.io_.dat.constants import PAD_ID_TAG, PAD_ID_HEADS

from camembert.downstream.finetune.io_.bert_iterators_tools.get_string_from_bpe import get_prediction, get_bpe_string, get_detokenized_str
from camembert.downstream.finetune.io_.get_new_batcher import get_new_shard, load_batcher_shard_data

from camembert.downstream.finetune.trainer.tools.epoch_run_fine_tuning_tools import get_casing,  log_warning, log_data_src_label_pred, tensorboard_loss_writer_batch_level, tensorboard_loss_writer_batch_level_multi, tensorboard_loss_writer_epoch_level,  writing_predictions_conll_multi, init_score_token_sent_dict, dimension_check_label,  tensorboard_loss_writer_epoch_level_multi, update_loss_dic_average, count_tokens, loss_mean
from camembert.downstream.finetune.trainer.tools.epoch_run_fine_tuning_tools import get_task_name_based_on_logit_label

from camembert.downstream.finetune.evaluate.report.report_writing import report_score_all
from camembert.downstream.finetune.evaluate.score.report import overall_word_level_metric_measure

from camembert.downstream.finetune.model.settings import TASKS_PARAMETER, LABEL_PARAMETER, SAMPLES_PER_TASK_TO_REPORT
from camembert.downstream.finetune.model.loss.get_losses import get_loss_multitask, get_penalization
from camembert.downstream.finetune.model.optimization.masking import dropout_input_tensor
from camembert.downstream.finetune.model.optimization.get_optmizers import apply_fine_tuning_strategy


def accumulate_scores_across_sents(agg_func_ls, sample_ls, dic_prediction_score, score_dic, n_tokens_dic, n_sents_dic):

    for agg_func in agg_func_ls:
        for sample in sample_ls:
            try:
                score_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["score"]
                n_tokens_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_tokens"]
                n_sents_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_sents"]
            except:
                print("ERROR accumulation score on {} sample {} agg".format(sample, agg_func))
                pdb.set_trace()
    return score_dic, n_tokens_dic, n_sents_dic


def epoch_run(batchIter, tokenizer,
              args,
              iter, n_obs_max, model, epoch,
              use_gpu, data_label, null_token_index, null_str,
              model_id, early_stoppin_metric=None,reader=None,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              args_dir=None,
              reference_word_dic=None, dropout_input_bpe=0.,
              writing_pred=False, dir_end_pred=None, extra_label_for_prediction="",
              log_perf=True, remove_mask_str_prediction=False,
              norm_2_noise_eval=False,
              compute_intersection_score=False,
              subsample_early_stoping_metric_val=None,
              threshold_edit=None,
              ponderation_loss_policy="static",
              samples_per_task_reporting=None,
              task_to_eval=None, task_to_label_dictionary=None,
              data_sharded_dir=None, n_shards=None, n_sent_dataset_total=None, args_load_batcher_shard_data=None,
              memory_efficient_iterator=False, model_origin=None,pruning_mask=None,  scheduler=None,
              verbose=0):
    """
    About Evaluation :
    Logic : compare gold and prediction topk using a word level scoring fucntion
            then accumulates for each sentences and foea each batch to get global score
            CAN add SAMPLE Parameter to get scores on specific subsample of the data : e.g. NEED_NORM, NORMED...
            Can also have different aggregation function
    """
    if optimizer is not None:
        # we need it to track distance in all cases
        # if training mode and penalize we need mode_origin
        assert model_origin is not None
    assert task_to_label_dictionary is not None, "ERROR : task_to_label_dictionary should be defined "

    if samples_per_task_reporting is None:
        samples_per_task_reporting = SAMPLES_PER_TASK_TO_REPORT
    if task_to_eval is not None:
        args.tasks = task_to_eval
        assert task_to_eval in task_to_label_dictionary, "ERROR : {} label was not provided in {}".format(task_to_eval, task_to_label_dictionary)
        printing("WARNING : task_to_eval was provided ", verbose=verbose, verbose_level=1)
    if ponderation_loss_policy == "static":
        assert args.multi_task_loss_ponderation is not None
    else:
        raise(Exception("Only static strategy supported so far"))

    if args.low_memory_foot_print_batch_mode:
        assert args.batch_update_train > 0, "ERROR have to define batch_size_real in low_memory_foot_print_batch_mode"

    if args.heuristic_ls is not None:
        for edit_rule in ["all", "ref", "data"]:
            if "edit_check-"+edit_rule in args.heuristic_ls:
                assert threshold_edit is not None, "ERROR threshold_edit required as args.heuristic_ls is {}".format(args.heuristic_ls)

    if args.case is not None:
        AVAILABLE_CASE_OPTIONS = ["lower"]
        assert args.case in AVAILABLE_CASE_OPTIONS
    assert args.norm_2_noise_training is None or not norm_2_noise_eval, "only one of the two should be triggered but we have args.norm_2_noise_training : {} norm_2_noise_eval:{}".format(args.norm_2_noise_training, norm_2_noise_eval)
    if args.norm_2_noise_training is not None:
        printing("WARNING : {} args.norm_2_noise_training is on ", var=[args.norm_2_noise_training],
                 verbose=verbose, verbose_level=1)
    if norm_2_noise_eval:
        printing("WARNING : {} norm_2_noise_eval is on ", var=[norm_2_noise_eval],
                 verbose=verbose, verbose_level=1)
    assert len(args.tasks) <= 2
    evaluated_task = []
    skip_score = 0
    skipping = 0
    mean_end_pred = 0
    label_heuristic = ""
    if memory_efficient_iterator:
        assert data_sharded_dir is not None and n_shards is not None, "ERROR data_sharded_dir and n_shards needed as args.memory_efficient_iterator {}".format(memory_efficient_iterator)
        assert n_sent_dataset_total is not None
    printing("INFO : HEURISTIC used {} {}", var=[args.heuristic_ls, label_heuristic], verbose=verbose, verbose_level=1)
    if predict_mode:
        if topk is None:
            topk = 1
            printing("PREDICTION MODE : setting top-k to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = False
        if metric is None:
            metric = "exact_match"
            printing("PREDICTION MODE : setting metric to default 'exact_match' ", verbose_level=1, verbose=verbose)

    if writing_pred:
        assert dir_end_pred is not None
        if extra_label_for_prediction != "":
            extra_label_for_prediction = "-"+extra_label_for_prediction
        extra_label_for_prediction += "-"+label_heuristic
        dir_normalized = os.path.join(dir_end_pred, "{}_ep-prediction{}.conll".format(epoch,
                                                                                      extra_label_for_prediction))
        dir_normalized_original_only = os.path.join(dir_end_pred, "{}_ep-prediction_src{}.conll".format(epoch,
                                                                                                        extra_label_for_prediction))
        dir_gold = os.path.join(dir_end_pred, "{}_ep-gold-{}.conll".format(epoch,
                                                                          extra_label_for_prediction))
        dir_gold_original_only = os.path.join(dir_end_pred, "{}_ep-gold_src{}.conll".format(epoch,
                                                                                            extra_label_for_prediction))

    mask_token_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    cls_token_index = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_token_index = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_token_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    #space_token_index = tokenizer.convert_tokens_to_ids([null_str])[0]
    printing("ITERATOR : {} : {} {} : {} {} : {} {} : {}", var=[tokenizer.mask_token, mask_token_index, tokenizer.cls_token, cls_token_index, tokenizer.sep_token, sep_token_index, tokenizer.pad_token, pad_token_index],
             verbose=verbose, verbose_level=1)
    printing("ITERATOR : PAD TAG {} PAD HEADS {}", var=[PAD_ID_TAG, PAD_ID_HEADS], verbose=verbose, verbose_level=1)
    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0
    n_obs_forwarded = 0
    n_obs_backward = 0
    n_obs_backward_save = 0
    n_obs_forwarded_not_backwarded = 0
    backprop_step = 0
    loss = 0
    penalize = 0
    penalization_dic = None
    report_penalization = False

    agg_func_ls = ["sum"]
    printout_allocated_gpu_memory(verbose=verbose, comment="starting epoch")
    score_dic, n_tokens_dic, n_sents_dic = init_score_token_sent_dict(samples_per_task_reporting, [task for tasks in args.tasks for task in tasks],
                                                                      agg_func_ls, compute_intersection_score,
                                                                      task_settings=TASKS_PARAMETER)

    _samples_per_task_reporting = list(samples_per_task_reporting.keys())+["all"]

    n_tokens_counter_per_task = OrderedDict((a, 0) for a in _samples_per_task_reporting)

    loss_dic_epoch = OrderedDict((a, 0) for a in _samples_per_task_reporting)
    # vocab_index_except_pad_cls_sep = [i for i in range(1, len(tokenizer.vocab)) if i not in [mask_token_index, sep_token_index, cls_token_index]]
    # pad is the first index
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True
    loss_norm = 0
    loss_pos = 0
    loss_n_mask_prediction = 0
    n_batch_pos = 0
    n_batch_norm = 0
    n_task_normalize_sanity = 0

    counting_failure_parralel_bpe_batch = 0

    time_multitask_train = 0
    time_backprop = 0
    time_multitask_preprocess_1 = 0
    time_multitask_preprocess_2 = 0
    time_multitask_postprocess = 0
    time_score = 0
    time_penalize = 0
    time_write_pred = 0
    backprop_step_former = -1
    time_overall_pass = time.time()
    end_schedule_lr = 0

    n_shard = 0

    while True:
        try:
            if memory_efficient_iterator and n_obs_forwarded >= n_sent_dataset_total:
                printing("BREAKING ALL ITERATORS memory_efficient_iterator True (mode is {}  shard {} ending) ",
                         var=[mode, n_shard], verbose_level=1, verbose=1)
                break
            batch_i += 1

            time_multitask_preprocess_start = time.time()
            
            start_schedule = time.time()

            if args.schedule_lr is not None and optimizer is not None:
                assert args.optimizer != "AdamW", "ERROR schedule_lr not supported in AdamW"
                assert args.fine_tuning_strategy == "standart",\
                    "ERROR only fine_tuning_strategy standart supported in shedule mode but is {} ".format(args.fine_tuning_strategy)

                def get_schedule_lr(args, i_step):
                    warmup_init_lr = 0.0000001
                    assert isinstance(args.n_steps_warmup, int) and args.n_steps_warmup>0, "ERROR n_steps_warmup {} ".format(args.n_steps_warmup)
                    #args.n_steps_warmup = 100
                    lr_step = (args.lr - warmup_init_lr) / args.n_steps_warmup
                    if i_step < args.n_steps_warmup:
                        lr = warmup_init_lr + i_step * lr_step
                    else:
                        lr = args.lr * (args.n_steps_warmup / i_step)**0.5
                    print("UPDATING OPTIMIZER WITH LR {} based on {} step , step warmup {} lr_step {} backprop ({} warming up )".format(lr, i_step, args.n_steps_warmup, lr_step, i_step < args.n_steps_warmup))
                    return lr
                if backprop_step != backprop_step_former:
                    lr = get_schedule_lr(args, i_step=backprop_step+1)
                    backprop_step_former = backprop_step
                    writer.add_scalars("opt/lr-schedule",  {"lr_model-{}".format(model_id): lr}, backprop_step)
                    writer.add_scalars("opt/lr-schedule2", {"lr_model-{}".format(model_id): lr}, backprop_step)
                    _, optimizer = apply_fine_tuning_strategy(model=model, fine_tuning_strategy=args.fine_tuning_strategy,lr_init=lr, betas=(0.9, 0.99),weight_decay=args.weight_decay,epoch=epoch, verbose=verbose)
            end_schedule_lr += time.time()-start_schedule

            batch = batchIter.__next__()
            # Normalization task is handled seperately
            # case the batches if case is 'lower'
            batch = get_casing(args.case, batch, False, cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token)
            #n_task_normalize_sanity += int(task_normalize_is)
            # handling normalization input
            time_multitask_preprocess_start = time.time()
            printout_allocated_gpu_memory(verbose=verbose, comment="starting step")
            # added but are related to the old flow
            #batch_raw_input, norm2noise_bool, args.norm_2_noise_training = input_normalization_processing(False, batch, args.norm_2_noise_training, False)

            #input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
            #    get_indexes(batch_raw_input, tokenizer, verbose, use_gpu, word_norm_not_norm=None)
            #input_mask = get_mask_input(input_tokens_tensor, use_gpu)
            #print(batch.)

            head_masks, input_tokens_tensor, token_type_ids, label_per_task,\
            input_tokens_tensor_per_task, input_mask_per_task, cumulate_shift_sub_word = get_label_per_bpe(tasks=args.tasks, batch=batch,
                                                                                          pad_index=pad_token_index,
                                                                                          use_gpu=use_gpu, tasks_parameters=TASKS_PARAMETER,
                                                                                          input_alignement_with_raw=None,
                                                                                          input_tokens_tensor=None,
                                                                                          masking_strategy=args.masking_strategy,
                                                                                          vocab_len=BERT_MODEL_DIC[args.bert_model]["vocab_size"],#len(tokenizer.vocab)-2,
                                                                                          mask_token_index=mask_token_index,
                                                                                          sep_token_index=sep_token_index,
                                                                                          cls_token_index=cls_token_index,
                                                                                          dropout_input_bpe=dropout_input_bpe)


            # NB : token_type_ids not used in MultiTask (no needed, just use 0 everywhere )
            #dimension_check_label(label_per_task, input_tokens_tensor)
            time_multitask_preprocess_1 += time.time()-time_multitask_preprocess_start
            printout_allocated_gpu_memory(verbose=verbose, comment="got input/output")
            # NB : we use the aligned input with the

            _1_to_n_token = 0

            if n_obs_forwarded >= n_obs_max:# and not args.low_memory_foot_print_batch_mode) or (batch_i == n_iter_max * int(args.batch_update_train // args.batch_size) and args.low_memory_foot_print_batch_mode):
                print("BREAKING ITERATION model {} because {} n_obs_max reached  (n_obs_forwarded {})".format(model_id, n_obs_max, n_obs_forwarded))
                break
            if batch_i % 1000 == 0:
                printing("TRAINING : iteration finishing {} batch", var=[batch_i], verbose=verbose, verbose_level=1)
            if _1_to_n_token:
                skipping_batch_n_to_1 += _1_to_n_token
                #continue
            # sanity checking alignement
            # we consider only 1 sentence case
            #printing("CUDA SANITY CHECK input_tokens:{}  type:{}input_mask:{}  label:{}", var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda, output_tokens_tensor_aligned.is_cuda], verbose=verbose, verbose_level="cuda")
            # we have to recompute the mask based on aligned input

            assert args.masking_strategy is None, "ERROR : {} not supported in multitask mode ".format(args.masking_strategy)

            # multitask :
            time_multitask_preprocess_2_start = time.time()

            n_tokens_counter_per_task, n_tokens_counter_current_per_task, n_tokens_all = count_tokens([task for tasks in args.tasks for task in tasks],
                                                                                                      n_tokens_counter_per_task, label_per_task, LABEL_PARAMETER)
            n_tokens_counter_per_task["all"] += n_tokens_all

            time_multitask_preprocess_2 += time.time()-time_multitask_preprocess_2_start
            time_multitask_train_start = time.time()

            logits_dic, loss_dic, _ = model(input_tokens_tensor_per_task, token_type_ids=None, labels=label_per_task, head_masks=head_masks, attention_mask=input_mask_per_task)
            #pdb.set_trace()
            printout_allocated_gpu_memory(verbose=verbose, comment="feedforward done")
            # loss_dic_epoch is the sum over all the epoch (mean computed for reporting)
            loss_dic_epoch = update_loss_dic_average(loss_dic, loss_dic_epoch)
            loss_dic = loss_mean(loss_dic, n_tokens_counter_current_per_task)

            if predict_mode:
                predictions_topk_dic = get_prediction(logits_dic, topk=topk)
                printout_allocated_gpu_memory(verbose=verbose, comment="prediction done")
                time_multitask_train += time.time()-time_multitask_train_start
                time_multitask_postprocess_start = time.time()
                assert "normalize" not in args.tasks, "ERROR : following line () was needed apparently for normalize being supported"
                # for parsing heads will leave heads untouched
                # POSTPROCESSING : get back to untokenized string
                source_preprocessed_dict, label_dic, predict_dic = get_bpe_string(predictions_topk_dic, label_per_task,
                                                                                  input_tokens_tensor_per_task, topk,
                                                                                  tokenizer, task_to_label_dictionary, #null_str, null_token_index,
                                                                                  TASKS_PARAMETER, mask_token_index, verbose)
                # input_tokens_tensor=eval("batch.{}".format(tasks_parameters[task]["input"])).clone(),
                # input_alignement_with_raw=eval("batch.{}_alignement".format(tasks_parameters[task]["input"]))
                #pdb.set_trace()
                #print("source", source_preprocessed_dict)

                src_detokenized_dic, label_detokenized_dic, predict_detokenize_dic = get_detokenized_str(source_preprocessed_dict=source_preprocessed_dict,
                                                                                                         input_alignement_with_raw=eval("batch.{}_alignement".format(TASKS_PARAMETER["pos"]["input"])),
                                                                                                         label_dic=label_dic, predict_dic=predict_dic,
                                                                                                         #remove_mask_str_prediction=remove_mask_str_prediction, null_str=null_str,
                                                                                                         # batch=batch,
                                                                                                         task_settings=TASKS_PARAMETER,
                                                                                                         flag_word_piece_token=BERT_MODEL_DIC[args.bert_model].get("wordpiece_flag",  "##"),
                                                                                                         flag_is_first_token=BERT_MODEL_DIC[args.bert_model].get("flag_is_first_token",0),
                                                                                                         # BERT_MODEL_DIC[args.bert_model].get("flag_is_first_token", 0),
                                                                                                         mask_str=tokenizer.mask_token, end_token=tokenizer.sep_token,
                                                                                                         cumulate_shift_sub_word=cumulate_shift_sub_word)

                #pdb.set_trace()
                if "parsing" in args.tasks[0]:
                    assert label_detokenized_dic["heads"]
                    try:
                        import numpy as np
                        for _ind, (gold_rebuild, gold) in enumerate(zip(label_detokenized_dic["heads"][0], batch.heads[0])):
                            if gold != -1:

                                assert gold_rebuild == gold, "ERROR {}: {} {},  " \
                                                             "label_detokenized_dic[heads] {} " \
                                                             "and batch.heads[0]) {}".format(_ind, gold_rebuild, gold,
                                                                                             label_detokenized_dic["heads"],
                                                                                             batch.heads[0])
                        print("VALIDATED")
                    except Exception as e:
                        print("HEADS SANITY CHECKED FAILED")
                        raise(e)
                        #pdb.set_trace()

                        #pdb.set_trace()
                #pdb.set_trace()
                log_data_src_label_pred(src_detokenized_dic, predict_detokenize_dic, label_detokenized_dic, tasks=args.tasks, verbose=verbose, verbose_level=2)
                printout_allocated_gpu_memory(verbose=verbose, comment="got string")
                label_processed = []
                time_multitask_postprocess += time.time() - time_multitask_postprocess_start
                # SCORING : get back to untokenized string
                time_score_start = time.time()
                for label_pred in predict_detokenize_dic:
                    label, _, _continue, label_processed = get_task_name_based_on_logit_label(label_pred, label_processed)
                    if _continue:
                        continue

                    task = re.match("(.*)-.*", label_pred).group(1)
                    src_detokenized = src_detokenized_dic[TASKS_PARAMETER[task]["input"]]
                    filter_score = samples_per_task_reporting[label_pred]
                    if label_detokenized_dic[label] is not None:
                        perf_prediction, skipping, _samples = overall_word_level_metric_measure(task_label=label, pred_label=label_pred,
                                                                                            gold_sent_ls_dict=label_detokenized_dic,
                                                                                            pred_sent_ls_topk_dict=predict_detokenize_dic,
                                                                                            topk=topk,
                                                                                            metric=metric,
                                                                                            samples=filter_score,
                                                                                            agg_func_ls=agg_func_ls,
                                                                                            reference_word_dic=reference_word_dic,
                                                                                            compute_intersection_score=compute_intersection_score,
                                                                                            mask_token=tokenizer.mask_token,
                                                                                            cls_token=tokenizer.cls_token,
                                                                                            sep_token=tokenizer.sep_token,
                                                                                            src_detokenized=src_detokenized)
                    else:
                        perf_prediction = {"score": 0,
                                           "agg_func": "sum", "metric": "exact_match",
                                           "n_tokens": 0,
                                           "n_sents": 0,}
                        skipping = 0
                        _samples = ["all"]

                    printing("PREDICTION epoch {} task {} score all {}/{} total "
                             "gold {} gold token {} pred {} pred token {} ",
                             var=[epoch, label, perf_prediction["sum"]["all"]["score"],
                                  perf_prediction["sum"]["all"]["n_tokens"],
                                  label_detokenized_dic[label], label_per_task[label],
                                  predict_detokenize_dic[label_pred], predictions_topk_dic[label_pred][:, :, 0]],
                             verbose=verbose, verbose_level="pred")

                    score_dic[label_pred], n_tokens_dic[label_pred], n_sents_dic[label_pred] = \
                        accumulate_scores_across_sents(agg_func_ls=agg_func_ls,
                                                       sample_ls=_samples, dic_prediction_score=perf_prediction,
                                                       score_dic=score_dic[label_pred],
                                                       n_tokens_dic=n_tokens_dic[label_pred],
                                                       n_sents_dic=n_sents_dic[label_pred])

                    evaluated_task.append(label_pred)
                # WRITTING PREDICTION
                time_score += time.time()-time_score_start
                time_write_pred_start = time.time()
                if writing_pred:
                    #batch_i#
                    # get the righ index sentence + handle batch size >1
                    # reader[task][0][0][-1][batch_i]
                    batch_sze = len(batch.raw_input)
                    raw_tests = []
                    sent_ids = []
                    append_mwe_ind = []
                    append_mwe_row = []
                    empty_mwe = True
                    for i_sent in range(batch_sze):
                        _append_mwe_row = []
                        _append_mwe_ind = []
                        # get raw sentence and idnex for the batch
                        comment_sent_i = reader[task][0][0][-1][batch_i-1+i_sent][-1]
                        raw_tests.append(comment_sent_i[0])
                        sent_ids.append(comment_sent_i[1])
                        # look for mwe
                        for word in reader[task][0][0][-1][batch_i-1+i_sent][0]:
                            if "-" in word[0]:
                                _append_mwe_row.append("\t".join(word)+"\n")
                                _append_mwe_ind.append(int(word[0].split("-")[0]))
                                empty_mwe = False
                        append_mwe_row.append(_append_mwe_row)
                        append_mwe_ind.append(_append_mwe_ind)
                    if empty_mwe:
                        append_mwe_row = None
                        append_mwe_ind = None
                    #pdb.set_trace()
                    new_file = writing_predictions_conll_multi(
                                            dir_pred=dir_normalized,
                                            append_mwe_ind=append_mwe_ind,
                                            append_mwe_row=append_mwe_row,
                                            sent_ids=sent_ids, raw_texts=raw_tests,
                                            dir_normalized_original_only=dir_normalized_original_only,
                                            dir_gold=dir_gold, dir_gold_original_only=dir_gold_original_only,
                                            src_detokenized=src_detokenized_dic, pred_per_task=predict_detokenize_dic,
                                            iter=iter, batch_i=batch_i, new_file=new_file, gold_per_tasks=label_detokenized_dic,
                                            all_indexes=batch.all_indexes, task_parameters=TASKS_PARAMETER,
                                            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token,
                                            tasks=args.tasks, verbose=verbose)
                time_write_pred += time.time() - time_write_pred_start
                printout_allocated_gpu_memory(verbose=verbose, comment="got score")
            report_penalization = optimizer is not None or (optimizer is None and epoch == 0)

            if report_penalization and args.ponderation_per_layer is not None:
                # NB : report_penalize is required if want to optimize using penalization
                time_get_penalize = time.time()
                penalize, penalization_dic = get_penalization(norm_order_per_layer=args.norm_order_per_layer,
                                                              ponderation_per_layer=args.ponderation_per_layer,
                                                              model_parameters=model.named_parameters(),
                                                              model_parameters_0=model_origin,
                                                              penalization_mode=args.penalization_mode,
                                                              pruning_mask=pruning_mask)

                printout_allocated_gpu_memory(verbose=verbose, comment="got penalization")

                if not args.penalize:
                    penalize = 0
                time_penalize += time.time() - time_get_penalize

            _loss = get_loss_multitask(loss_dic, args.multi_task_loss_ponderation)
            loss_dic["multitask"] = _loss.detach().clone().cpu()
            _loss += penalize
            loss_dic["all"] = _loss
            # temporary
            # based on a policy : handle batch, epoch, batch weights, simultanuously
            # assert the policy is consistent with the available labels fed to the model
            # training :
            time_backprop_start = time.time()
            loss += _loss.detach()
            # BACKWARD PASS
            # batch_i is the iteration counter
            #back_pass = optimizer is not None and ((args.low_memory_foot_print_batch_mode and batch_i % int(args.batch_update_train // args.batch_size) == 0) or not args.low_memory_foot_print_batch_mode)
            back_pass = optimizer is not None and (args.low_memory_foot_print_batch_mode and n_obs_forwarded >= args.batch_update_train) or not args.low_memory_foot_print_batch_mode
            n_obs_forwarded_not_backwarded += input_tokens_tensor_per_task[list(input_tokens_tensor_per_task.keys())[0]].size(0)
            n_obs_forwarded += input_tokens_tensor_per_task[list(input_tokens_tensor_per_task.keys())[0]].size(0)
            if optimizer is not None:
                mode = "train"

                _loss.backward()

                printout_allocated_gpu_memory(verbose, "loss backwarded")
                if (args.low_memory_foot_print_batch_mode and n_obs_forwarded >= args.batch_update_train) or not args.low_memory_foot_print_batch_mode:
                    n_obs_backward = n_obs_forwarded_not_backwarded
                    n_obs_backward_save += n_obs_forwarded_not_backwarded
                    n_obs_forwarded_not_backwarded = 0
                    backprop_step += 1
                    if args.low_memory_foot_print_batch_mode:
                        printing("OPTIMIZING in low_memory_foot_print_batch_mode cause batch index {}"
                                 "we update every {} batch_update_train {} batch_size to get batch backward pass of size {}",
                                 var=[batch_i, args.batch_update_train, args.batch_size, args.batch_update_train],
                                 verbose=verbose, verbose_level=1)
                    for opti in optimizer:
                        opti.step()
                        if scheduler is not None:
                            printing("OPTIMIZING : updating scheduler current step {} backward pass {} lr {}  init lr {}",
                                     var=[batch_i, backprop_step, opti.param_groups[0]["lr"],
                                          opti.param_groups[0]["initial_lr"]],
                                     verbose=verbose, verbose_level=1)
                            scheduler.step()

                        opti.zero_grad()
                printout_allocated_gpu_memory(verbose=verbose, comment="optimized")
            else:
                mode = "dev"
            time_backprop += time.time() - time_backprop_start

            #if writer is not None:
            #    tensorboard_loss_writer_batch_level(writer, mode, model_id, _loss, batch_i, iter,  loss_dic,
            #                                        False, args.append_n_mask)
            #    tensorboard_loss_writer_batch_level_multi(writer, mode, model_id, _loss, batch_i, iter, loss_dic, tasks=args.tasks)

        except StopIteration:
            printing("BREAKING ITERATION model {} -  {} iter - {} n_obs_forwarded  -  {} n_obs_backward or {} step of {} batch_update_train"
                     "(mode is {} memory_efficient_iterator {} , shard {} ending ",
                     var=[model_id, batch_i, n_obs_forwarded, n_obs_backward_save, backprop_step, args.batch_update_train, mode, memory_efficient_iterator, n_shard],
                     verbose_level=1, verbose=1)
            if optimizer is not None:
                assert n_obs_backward > 0, "ERROR : train mode but did not backpropagage any thing "
            n_shard += 1
            if not memory_efficient_iterator:
                break
            training_file = get_new_shard(data_sharded_dir, n_shards, verbose=verbose)
            printing("ITERATOR shard model {} - epoch {} , n observations forwarded {} "
                     "batch {} (n_ob_max {}) starting new {} ".format(model_id, epoch, n_obs_forwarded, batch_i, n_obs_max, training_file),
                     verbose=verbose, verbose_level=1)
            batchIter = load_batcher_shard_data(args, args_load_batcher_shard_data, training_file, verbose)

    overall_pass = time.time()-time_overall_pass

    printing("TIME epoch {}/{} done mode_id {}  {:0.3f}/{:0.3f} min "
             "({} forward obs {} backward obs) for {} iteration of {} batch_size in {} (fixed) averaged if flexible {} "
             " mode out of {} sent total or {} steps : {} min/batch {} min/sent",
             var=[epoch, args.epochs,
                  model_id,
                  overall_pass/60,n_sent_dataset_total * (overall_pass / 60) / n_obs_forwarded if n_sent_dataset_total is not None else 0,
                  n_obs_backward, n_obs_forwarded,
                  batch_i, args.batch_size, args.batch_size,
                  "train" if optimizer is not None else "dev", str(n_sent_dataset_total),
                  n_sent_dataset_total / args.batch_size if n_sent_dataset_total is not None else "_",
                  overall_pass/60/batch_i, overall_pass/60/n_obs_forwarded ],
             verbose_level=1, verbose=verbose)

    timing = OrderedDict([("time_multitask_preprocess_1 (get_label)", "{:0.4f} min/total {:0.4f} s/batch".format(time_multitask_preprocess_1/60, time_multitask_preprocess_1/batch_i)),
             ("time_multitask_preprocess_2 (count)", "{:0.4f} min/total {:0.4f} s/batch".format(time_multitask_preprocess_2/60, time_multitask_preprocess_2/batch_i)),
             ("time_multitask_feedforward (foward+pred)", "{:0.4f} min/total {:0.4f} s/batch".format(time_multitask_train/60, time_multitask_train/batch_i)),
             ("time_penalize", "{:0.4f} min/total {:0.4f} s/batch".format(time_penalize/60, time_penalize/batch_i)),
             ("time_multitask_backprop","{:0.4f} min/total {:0.4f} s/batch".format(time_backprop / 60, time_backprop / batch_i)),
             ("time_write_pred", "{:0.4f} min/total {:0.4f} s/batch".format(time_write_pred / 60, time_write_pred / batch_i)),
             ("time_multitask_get_string (get string) ", "{:0.4f} min/total {:0.4f} s/batch".format(time_multitask_postprocess/60, time_multitask_postprocess/batch_i)),
             ("time_score (score) ","{:0.4f} min/total {:0.4f} s/batch".format(time_score / 60, time_score / batch_i)),
             ("time schedule lr ", "scheduleer {:0.4} min in average".format(end_schedule_lr/batch_i)),
             ])

    print("TIME epoch {}/{} ({} step of {} size in {} mode {} pass (done mode_id {}  task:{}): {})".format(epoch, args.epochs,
                                                                                                           batch_i,
                                                                                                           args.batch_size,
                                                                                                           "predict" if optimizer is None else "train/accumulate",
                                                                                                           "backward" if back_pass else "foward",
                                                                                                           model_id,
                                                                                                           args.tasks,
                                                                                                           timing))
    log_warning(counting_failure_parralel_bpe_batch, data_label, batch_i, batch, noisy_under_splitted, skipping_batch_n_to_1, aligned, noisy_over_splitted, skip_1_t_n, skipping_evaluated_batch, verbose)

    early_stoppin_metric_val = 999
    evaluated_task = list(set(evaluated_task))
    if predict_mode:
        if writer is not None:
            # n_tokens_counter_per_task
            tensorboard_loss_writer_epoch_level_multi(writer,  mode, model_id, epoch, loss_dic_epoch,
                                                      n_tokens_counter_per_task, data_label,
                                                      penalization_dic=penalization_dic if report_penalization else None,
                                                      group_mapping=["bert.encoder.layer.*.attention.*", "bert.encoder.layer.*.intermediate.*",
                                                                     "bert.encoder.layer.*.output.*", "bert.embedding", "bert.pooler", "head"])

            tensorboard_loss_writer_epoch_level(writer, args.tasks, mode, model_id, epoch, n_batch_norm, n_batch_pos, args.append_n_mask, loss, loss_norm, loss_pos, loss_n_mask_prediction, batch_i, data_label)
        printing("TRAINING : evaluating on {} args.tasks ", var=[evaluated_task], verbose_level=1, verbose=verbose)
        reports = []
        reports, early_stoppin_metric_val, score, n_tokens = report_score_all(evaluated_task, agg_func_ls, samples_per_task_reporting, label_heuristic, score_dic, n_tokens_dic, n_sents_dic, model_id, args.tasks, args_dir,
                                                                              data_label, reports,  writer, log_perf, early_stoppin_metric_val,
                                                                              early_stoppin_metric, mode, subsample_early_stoping_metric_val, epoch)
    else:
        reports = None
    iter += batch_i
    if writing_pred:
        printing("DATA WRITTEN TO {} ", var=[dir_end_pred], verbose=verbose, verbose_level=1)
    printing("END EPOCH {} mode, iterated {} on normalisation ", var=[mode, n_task_normalize_sanity], verbose_level=1, verbose=verbose)

    # eval NER :

    if writing_pred:
        printing("SCORE computing F1 ",
                 verbose=verbose, verbose_level=1)
        f1 = evaluate(dataset_name=None,
                      dataset=None,
                      dir_end_pred=dir_end_pred,
                      prediction_file=dir_normalized, #os.path.join(dir_end_pred, "LAST_ep-prediction-fr_ftb_pos_ner-ud-test-.conll"),
                      gold_file_name=dir_gold) #os.path.join(dir_end_pred, "LAST_ep-gold--fr_ftb_pos_ner-ud-test-.conll"))
        if f1 is not None:
            f1 = f1/100
        f1_report = report_template(metric_val="f1", info_score_val="all",score_val=f1,model_full_name_val=model_id,
                                    report_path_val=None, evaluation_script_val="conlleval",
                                    data_val=data_label,
                                    model_args_dir=args_dir,
                                    n_tokens_score=None, task=args.tasks, n_sents=None, token_type="word", subsample="all",
                                    avg_per_sent=None, min_per_epoch=None)
        if predict_mode:
            reports.append(f1_report)
        if "ftb" in data_label:
            printing("WARNING : defining early_stoppin_metric_val based on F1 ", verbose=verbose, verbose_level=1)
            early_stoppin_metric_val = -f1
            printing("SCORE model {} data {} epoch {} NER : score {}", var=[model_id, data_label, epoch, f1], verbose=verbose, verbose_level=1)



    try:
        if early_stoppin_metric is not None:
            assert early_stoppin_metric_val is not None, "ERROR : early_stoppin_metric_val should have been found " \
                                                         "but was not {} sample metric {}  not found in {} (NB : MIGHT ALSO BECAUSE THE PERF DID NOT DECREASED AT ALL ) ".format(early_stoppin_metric, subsample_early_stoping_metric_val, reports)
    except Exception as e:
        print(e)
    if early_stoppin_metric_val is None:
        print("WARNING : early_stoppin_metric_val is None, score {} n_tokens {}".format(score, n_tokens))
    return loss/batch_i, iter, reports, early_stoppin_metric_val
