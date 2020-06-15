
from camembert_finetune.env.imports import pdb, torch, OrderedDict, os, tqdm
from camembert_finetune.args.args_parse import args_predict, args_check
from camembert_finetune.io_.bert_iterators_tools.get_string_from_bpe import get_prediction, get_bpe_string, get_detokenized_str
from camembert_finetune.predict.prediction_pipe import load_tok_model_for_prediction, detokenized_src_label
from camembert_finetune.model.settings import TASKS_PARAMETER
from camembert_finetune.io_.data_iterator import readers_load, data_gen_multi_task_sampling_batch
from camembert_finetune.io_.bert_iterators_tools.get_bpe_labels import get_label_per_bpe
from camembert_finetune.evaluate.eval_ner import evaluate


def predict(args):

    model, tokenizer, task_to_label_dictionary, dictionaries, vocab_size = load_tok_model_for_prediction(args)

    if args.mode == "interactive":
        model.eval()

        while True:
            text = input(f"Type a sentence and Camembert will process it (or 'exit/q') with task {args.task} \nNB : Space each word (e.g. l' Italie) \nYou --> ")
            if text in ["exit", "q"]:
                print("Exiting interaction with Camembert")
                break
            encoded = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            inputs = OrderedDict([("wordpieces_inputs_words", encoded)])
            attention_mask = OrderedDict(
                [("wordpieces_inputs_words",  torch.ones_like(encoded))])

            logits, _, _ = model(input_ids_dict=inputs, attention_mask=attention_mask)

            predictions_topk = get_prediction(logits, topk=1)
            label_ls = TASKS_PARAMETER[args.tasks[0][0]]["label"]
            label = OrderedDict([(label, None) for label in label_ls])

            source_preprocessed, label_dic, predict_dic = get_bpe_string(predictions_topk_dic=predictions_topk,
                                                                         output_tokens_tensor_aligned_dic=label,
                                                                         input_tokens_tensor_per_task=inputs, topk=1,
                                                                         tokenizer=tokenizer,
                                                                         task_to_label_dictionary=task_to_label_dictionary,
                                                                         task_settings=TASKS_PARAMETER, mask_index=tokenizer.mask_token_id,
                                                                         verbose=1)

            detokenized_source_preprocessed, detokenized_label_batch, _ = detokenized_src_label(source_preprocessed,predict_dic, label_ls)

            # ASSUME BATCH len 1
            if args.output_tokenized:
                print(f"Camembert tokenized input {source_preprocessed['wordpieces_inputs_words'][0][1:-1]}")
                for label, pred_label in zip(label_ls, predict_dic):
                    print(f"Camembert {label if args.task!='ner' else 'NER'} prediction: {predict_dic[pred_label][0][0][1:-1]}")

            print(f"You --> {detokenized_source_preprocessed['wordpieces_inputs_words'][0]}")
            for label, pred_label in zip(label_ls, predict_dic):
                print(f"Camembert {label.upper() if args.task != 'ner' else 'NER'} --> {detokenized_label_batch[pred_label][0]}\n")

    elif args.mode == "conll":
        # loading conll filed based on the task we want
        readers = readers_load(datasets=args.input_file,  # if not args.memory_efficient_iterator else training_file,
                               tasks=args.tasks, word_dictionary=dictionaries["word"],
                               bert_tokenizer=tokenizer,
                               word_dictionary_norm=dictionaries["word_norm"],
                               char_dictionary=dictionaries["char"],
                               pos_dictionary=dictionaries["pos"], xpos_dictionary=dictionaries["xpos"],
                               type_dictionary=dictionaries["type"],
                               word_decoder=True, run_mode="test",
                               add_start_char=1, add_end_char=1, symbolic_end=1,
                               symbolic_root=1, bucket=False,
                               must_get_norm=True, input_level_ls=["wordpiece"], verbose=1)
        # creating batch iterator

        batchIter = data_gen_multi_task_sampling_batch(tasks=args.tasks,
                                                       readers=readers,
                                                       batch_size=args.batch_size,  # readers[list(readers.keys())[0]][4],
                                                       max_token_per_batch=None,
                                                       # ,max_token_per_batch if flexible_batch_size else None,
                                                       word_dictionary_norm=dictionaries["word_norm"],
                                                       word_dictionary=dictionaries["word"],
                                                       char_dictionary=dictionaries["char"],
                                                       pos_dictionary=dictionaries["pos"],
                                                       get_batch_mode=False,
                                                       print_raw=False,
                                                       dropout_input=0.0,
                                                       verbose=1)

        new_batch = 0
        task = args.tasks[0][0]

        pred_file = args.output_file
        new_id = False
        if os.path.isfile(pred_file):
            print(f"WARNING : {pred_file} exist : over-writing it")

        with open(pred_file, "w") as f:
            n_sent = readers[task][-2]
            n_iter = int(n_sent/args.batch_size)
            print(f"Starting prediction... processing {n_sent} sentences, batch size {args.batch_size} ")
            pbar = tqdm(total=n_iter)
            # iterating through file, predicting and writing prediction to pred_file
            while True:
                try:
                    batch = batchIter.__next__()
                    pbar.update(1)
                    new_batch += 1
                except StopIteration:
                    pbar.close()
                    f.write("\n")
                    break

                head_masks, input_tokens_tensor, token_type_ids, label_per_task, \
                input_tokens_tensor_per_task, input_mask_per_task, cumulate_shift_sub_word = \
                    get_label_per_bpe(tasks=args.tasks, batch=batch, pad_index=tokenizer.pad_token_id, use_gpu=False,
                                      tasks_parameters=TASKS_PARAMETER, input_alignement_with_raw=None,
                                      input_tokens_tensor=None, masking_strategy=None,
                                      vocab_len=vocab_size, mask_token_index=tokenizer.mask_token_id,
                                      sep_token_index=tokenizer.eos_token_id, cls_token_index=tokenizer.bos_token_id,
                                      dropout_input_bpe=0.0)

                logits_dic, loss_dic, _ = model(input_tokens_tensor_per_task, token_type_ids=None,
                                                labels=label_per_task, head_masks=head_masks,
                                                attention_mask=input_mask_per_task)

                predictions_topk = get_prediction(logits_dic, topk=1)
                label_ls = TASKS_PARAMETER[args.tasks[0][0]]["label"]

                source_preprocessed, label_dic, predict_dic = get_bpe_string(predictions_topk_dic=predictions_topk,
                                                                             output_tokens_tensor_aligned_dic=label_per_task,
                                                                             input_tokens_tensor_per_task=input_tokens_tensor_per_task,
                                                                             topk=1,
                                                                             tokenizer=tokenizer,
                                                                             task_to_label_dictionary=task_to_label_dictionary,
                                                                             task_settings=TASKS_PARAMETER,
                                                                             mask_index=tokenizer.mask_token_id,
                                                                             verbose=1)

                detokenized_source_preprocessed, detokenized_label_batch, gold_label_batch = detokenized_src_label(
                                                source_preprocessed, predict_dic, label_ls, label_dic=label_dic)

                ind = 0

                ls_key = list(detokenized_label_batch.keys())
                n_key = len(ls_key)
                first_key = ls_key[0]

                def get_mwe_ind_row():
                    batch_sze = len(batch.raw_input)
                    task = args.tasks[0][0]
                    append_mwe_ind = []
                    append_mwe_row = []
                    raw_tests = []
                    sent_ids = []
                    empty_mwe = True

                    for i_sent in range(batch_sze):
                        _append_mwe_row = []
                        _append_mwe_ind = []
                        # get raw sentence and idnex for the batch

                        comment_sent_i = readers[task][0][1][-1][new_batch - 1 + i_sent][-1]
                        raw_tests.append(comment_sent_i[1])
                        # sent_ids.append(comment_sent_i[1])
                        # look for mwe
                        mwe = False
                        if mwe:
                            for word in readers[task][0][1][-1][new_batch - 1 + i_sent][0]:
                                if "-" in word[0]:
                                    _append_mwe_row.append("\t".join(word) + "\n")
                                    _append_mwe_ind.append(int(word[0].split("-")[0]))
                                    empty_mwe = False
                            append_mwe_row.append(_append_mwe_row)
                            append_mwe_ind.append(_append_mwe_ind)
                    if empty_mwe:
                        append_mwe_row = None
                        append_mwe_ind = None
                    return append_mwe_row, append_mwe_ind, sent_ids, raw_tests

                # append_mwe_row, append_mwe_ind, sent_ids, raw_tests = get_mwe_ind_row()

                for batch in range(len(detokenized_source_preprocessed["wordpieces_inputs_words"])):

                    if (new_batch-1)*args.batch_size+batch+1 > 1:
                        f.write("\n")

                    if new_id:
                        f.write(f"# sent_id = {(new_batch-1)*args.batch_size + batch+1} \n")
                    else:
                        conll_comment_ls = readers[task][0][0][-1][(new_batch-1)*args.batch_size + batch][-1]
                        for conll_comment in conll_comment_ls:
                            f.write(conll_comment)

                    if n_key == 1:
                        for word, pred, gold in zip(detokenized_source_preprocessed["wordpieces_inputs_words"][batch],
                                                    detokenized_label_batch[first_key][batch],
                                                    gold_label_batch[first_key][batch]):
                            ind += 1
                            pos = "_"
                            head = "_"
                            dep = "_"
                            if args.task == "pos":
                                pos = pred
                            elif args.task == "ner":
                                pos = pred
                            else:
                                raise (Exception)
                            f.write("{ind}\t{word}\t_\t{pos}\t_\t_\t{head}\t{dep}\t_\t_\n".format(ind=ind, word=word,
                                                                                                  pos=pos, head=head,
                                                                                                  dep=dep))
                    else:
                        assert args.task == "parsing"
                        second_key = ls_key[1]
                        for word, pred, gold, pred_2, gold_2 in zip(
                                detokenized_source_preprocessed["wordpieces_inputs_words"][batch],
                                detokenized_label_batch[first_key][batch], gold_label_batch[first_key][batch],
                                detokenized_label_batch[second_key][batch], gold_label_batch[second_key][batch]):
                            ind += 1
                            pos = "_"
                            head = pred
                            dep = pred_2

                            f.write("{ind}\t{word}\t_\t{pos}\t_\t_\t{head}\t{dep}\t_\t_\n".format(ind=ind, word=word,
                                                                                                  pos=pos, head=head,
                                                                                                  dep=dep))
                    batch += 1
            print(f"Camembert prediction written {pred_file}")

        if args.gold_file is not None:
            print("Gold file provided so running evaluation")

            if args.task == "ner":

                print("Evaluating NER with official conlleval CoNLL-03 script")
                dir = os.path.dirname(os.path.abspath(__file__))
                pred_file = os.path.abspath(pred_file)
                args.gold_file = os.path.abspath(args.gold_file)
                f1 = evaluate(dir_end_pred=os.path.join(dir, "camembert_finetune/evaluate/eval_temp"),
                              prediction_file=pred_file,
                              gold_file_name=args.gold_file, verbose=2 if args.score_details else 1)
                print("Overall F1 score : ", f1)
            else:
                raise(Exception("not supported "))
        else:
            assert args.score_details, "ERROR : --gold_file should be provided if --score_details 1"


if "__main__" == __name__:

    args = args_predict()
    args_check(args)
    predict(args)

