from camembert_finetune.env.imports import pdb
from camembert_finetune.io_.logger import printing
from camembert_finetune.model.settings import LABEL_PARAMETER
from camembert_finetune.model.constants import NULL_STR, PADING_SYMBOLS


def align_bpe(n_bpe_target_minus_source, source_aligned, source_aligned_index, target_aligned, target_aligned_index,
              n_masks_to_add, src_token_len, bert_tokenizer, mask_token,
              mode="dummy", index_src=None, index_target=None, verbose=0):
    """
    align bpe of a given token using mode
    :return:
    """
    assert mode in ["dummy"]
    # dummy means appending with SPACE or MASK when needed
    if n_bpe_target_minus_source > 0:
        assert index_src is not None
        source_aligned_index.extend([index_src for _ in range(n_bpe_target_minus_source)])
        source_aligned.extend(
            bert_tokenizer.convert_tokens_to_ids([mask_token for _ in range(n_bpe_target_minus_source)]))

    elif n_bpe_target_minus_source < 0:
        assert index_target is not None
        # we add a NULL_STR (to be predicted) and index it as the former bpe token
        target_aligned_index.extend([index_target for _ in range(-n_bpe_target_minus_source)])
        target_aligned.extend(bert_tokenizer.convert_tokens_to_ids([NULL_STR for _ in range(-n_bpe_target_minus_source)]))

    n_masks_to_add.append(n_bpe_target_minus_source)
    n_masks_to_add.extend([-1 for _ in range(src_token_len - 1)])

    if verbose == "reader":
        printing("SRC appending word bpe align : {}\nTARGET appending word bpe align : {} \nN_MASKS------------ : {}",
                 var=[[mask_token for _ in range(n_bpe_target_minus_source)] if n_bpe_target_minus_source > 0 else "",
                      [NULL_STR for _ in range(-n_bpe_target_minus_source)] if n_bpe_target_minus_source < 0 else "",
                      [n_bpe_target_minus_source]+[-1 for _ in range(src_token_len - 1)]],
                 verbose_level="reader", verbose=verbose)

    return source_aligned, source_aligned_index, target_aligned, target_aligned_index, n_masks_to_add


def realigne_multi(ls_sent_str, input_alignement_with_raw,
                   mask_str, label,
                   end_token,
                   #remove_null_str=True, null_str,
                   remove_mask_str=False, remove_extra_predicted_token=False,
                   keep_mask=False, gold_sent=False, flag_word_piece_token="##", flag_is_first_token=False,
                   cumulate_shift_sub_word=None):
    """
    # factorize with net realign
    ** remove_extra_predicted_token used iif pred mode **
    - detokenization of ls_sent_str based on input_alignement_with_raw index
    - we remove paddding and end detokenization at symbol [SEP] that we take as the end of sentence signal
    """
    assert len(ls_sent_str) == len(input_alignement_with_raw), "ERROR : ls_sent_str {} : {} input_alignement_with_raw {}" \
                                                               " : {} ".format(ls_sent_str, len(ls_sent_str),
                                                                               input_alignement_with_raw,
                                                                               len(input_alignement_with_raw))
    new_sent_ls = []
    if label == "heads":
        assert cumulate_shift_sub_word is not None
    ind_sent = 0
    DEPLOY_MODE = True
    for sent, index_ls in zip(ls_sent_str, input_alignement_with_raw):
        # alignement index and input/label should have same len
        assert len(sent) == len(index_ls), "ERROR : {} sent {} len {} and index_ls {} len {} not same len".format(label, sent, index_ls, len(sent), len(index_ls))

        former_index = -1
        new_sent = []
        former_token = ""

        for _i, (token, index) in enumerate(zip(sent, index_ls)):

            trigger_end_sent = False
            index = int(index)

            if remove_extra_predicted_token:
                if index == 1000 or index == -1:
                    # we reach the end according to gold data
                    # (this means we just stop looking at the prediciton of the model (we can do that because we assumed word alignement))
                    trigger_end_sent = True
                    if gold_sent:
                        # we sanity check that the alignement corredponds
                        try:
                            assert token in PADING_SYMBOLS, "WARNING 123 : breaking gold sequence on {} token not in {}".format(token , PADING_SYMBOLS)
                        except Exception as e:
                            print(e)
            # if working with input : handling mask token in a specific way
            if token == mask_str and not keep_mask:
                token = "X" if not remove_mask_str else ""
            # if working with input merging # concatanating wordpieces
            if LABEL_PARAMETER[label]["realignement_mode"] == "detokenize_bpe":
                if index == former_index:
                    if token.startswith(flag_word_piece_token) and not flag_is_first_token:
                        former_token += token[len(flag_word_piece_token):]
                    else:
                        former_token += token
            # for sequence labelling : ignoring
            elif LABEL_PARAMETER[label]["realignement_mode"] == "ignore_non_first_bpe":
                # we just ignore bpe that are not first bpe of tokens
                if index == former_index:
                    pass
            # if new token --> do something on the label
            # if index != former_index or _i + 1 == len(index_ls): # for DEPLOY_MODE = False
            if (index != former_index or index == -1) or _i + 1 == len(index_ls):
                if not flag_is_first_token:
                    new_sent.append(former_token)
                elif flag_is_first_token and (isinstance(former_token, str) and former_token.startswith(flag_word_piece_token)):
                    new_sent.append(former_token[len(flag_word_piece_token):])
                else:
                    if label == "heads":
                        #print("WARNING : HEAD RE-ALIGNING")
                        if isinstance(former_token, int):
                            try:
                                #print(cumulate_shift_sub_word[ind_sent][former_index], former_token, cumulate_shift_sub_word[ind_sent][former_token])
                                #cumulate_shift_sub_word[ind_sent][former_token]
                                #print(ls_sent_str[ind_sent][former_index], former_token)
                                if former_token != -1:
                                    #pdb.set_trace()
                                    #print("-->",former_index)
                                    #former_token -= cumulate_shift_sub_word[ind_sent][former_token]
                                    former_token = eval(input_alignement_with_raw[ind_sent][former_token])
                                    token = eval(input_alignement_with_raw[ind_sent][token])
                            except:
                                print("error could not process former_token {} too long for cumulated_shift {} ".format(former_token, cumulate_shift_sub_word[ind_sent]))
                                if gold_sent:
                                    pdb.set_trace()
                            #former_token-=cumulate_shift_sub_word[ind_sent][former_token]
                    # is this last case possible


                    #new_sent.append(former_token)
                    new_sent.append(token)

                former_token = token
                if trigger_end_sent:
                    print("break trigger_end_sent")
                    break

            elif DEPLOY_MODE:
                # EXCEPT PUNCTUNATION FOR WHICH SHOULD ADD -1 BEFORE !

                #if former_index != -1:
                #    new_sent.append(eval(input_alignement_with_raw[ind_sent][former_token]))
                former_token = token
                new_sent.append(-1)

                # ADD MORE IF
            #pdb.set_trace()
            # if not pred mode : always not trigger_end_sent : True
            # (required for the model to not stop too early if predict SEP too soon)
            # NEW CLEANER WAY OF BREAKING : should be generalize
            if remove_extra_predicted_token and trigger_end_sent:
                if not flag_is_first_token:
                    new_sent.append(former_token)
                elif flag_is_first_token and (isinstance(former_token, str) and former_token.startswith(flag_word_piece_token)):
                    new_sent.append(former_token[len(flag_word_piece_token):])
                else:
                    # is this last case possible
                    new_sent.append(former_token)
                print("break remove_extra_predicted_token")
                break
            # TODO : SHOULD be cleaned
            # XLM (same first and end token) so not activated for </s>

            if not DEPLOY_MODE:
                if ((former_token == end_token and end_token != "</s>") or _i + 1 == len(index_ls) and not remove_extra_predicted_token) or ((remove_extra_predicted_token and (former_token == end_token and trigger_end_sent) or _i + 1 == len(index_ls))):
                    new_sent.append(token)

                    print(f"break new_sent {((former_token == end_token and end_token != '</s>') or _i + 1 == len(index_ls) and not remove_extra_predicted_token)} or { ((remove_extra_predicted_token and (former_token == end_token and trigger_end_sent) or _i + 1 == len(index_ls)))}")
                    break
            former_index = index
        #if DEPLOY_MODE:
        #    new_sent_ls.append(new_sent[1:])
        #else:
        #new_sent_ls.append(new_sent[1:])
        new_sent_ls.append(new_sent)
        ind_sent += 1
    if gold_sent:
        print("CUMULATED SHIFT", cumulate_shift_sub_word)
        print("GOLD:OUTPUT BEFORE DETOKENIZATION ", ls_sent_str)
        print("GOLD:OUTPUT AFTER DETOKENIZATION", new_sent_ls)

    return new_sent_ls

