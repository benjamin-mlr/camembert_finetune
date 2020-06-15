

from __future__ import division
from __future__ import print_function
import pdb
import argparse
import io
import os
import sys
import unittest
from copy import deepcopy
from collections import OrderedDict
# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# Content and functional relations
CONTENT_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep"
}

FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc"
}

UNIVERSAL_FEATURES = {
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
    "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
    "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
}


class UDError(Exception):
    pass
# Load given CoNLL-U file into internal representation


def load_conllu(file, raise_all=True):
    # Internal representation classes
    class UDRepresentation:
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.sentences = []
            self.fixing_sents = OrderedDict()
            self.root_per_sent = OrderedDict()

        def write(self, file):
            root_to_define = None
            fix_mode = "?"
            fixing_ls = []
            root_to_define_counter_fixed = 0
            for ind, word in enumerate(self.words):

                word.columns = [str(col) for col in word.columns]

                if ind in self.fixing_sents:
                    # means we are in a sentence that needs to be fixed
                    if "no_root_and_cycle" in self.fixing_sents[ind]:
                        assert ind not in fixing_ls, f"ERROR sent of ind {ind} already fixed "
                        fixing_ls.append(ind)
                        fix_mode = "no_root_and_cycle"
                        # we get the index of the word whose head needs to be fix
                        root_to_define = self.fixing_sents[ind]["no_root_and_cycle"]
                        point_to = str(0)
                    elif "1_root_cycle" in self.fixing_sents[ind]:
                        assert ind not in fixing_ls, f"ERROR sent of ind {ind} already fixed "
                        fixing_ls.append(ind)
                        #pdb.set_trace()
                        fix_mode = "1_root_cycle"
                        root_to_define = self.fixing_sents[ind]["1_root_cycle"]
                        point_to = self.root_per_sent[ind]
                    elif "several_root_no_cycle" in self.fixing_sents[ind]:
                        assert ind not in fixing_ls, f"ERROR sent of ind {ind} already fixed "
                        fixing_ls.append(ind)
                        fix_mode = "several_root_no_cycle"
                        root_to_define = self.fixing_sents[ind]["several_root_no_cycle"]
                        point_to = self.root_per_sent[ind]
                        root_to_define_counter_fixed = 0
                    elif "several_root_and_cycle" in self.fixing_sents[ind]:
                        assert ind not in fixing_ls, f"ERROR sent of ind {ind} already fixed "
                        fixing_ls.append(ind)
                        fix_mode = "several_root_and_cycle"
                        root_to_define = self.fixing_sents[ind]["several_root_and_cycle"]["root"] # it is a list
                        root_to_define.append(self.fixing_sents[ind]["several_root_and_cycle"]["cycle"])
                        point_to = self.root_per_sent[ind]
                        root_to_define_counter_fixed = 0

                if word.columns[ID] == "1":
                    if ind > 0:
                        file.write("\n")
                    file.write("# new sent \n")
                if root_to_define is not None:
                    #pdb.set_trace()
                    if isinstance(root_to_define, str):
                        if word.columns[ID] == root_to_define:
                            print(f"HEURISTIC : writing : Word of index {word.columns[ID]} with head {word.columns[HEAD]} because {fix_mode} set to root {point_to}")
                            fix_mode = "?"
                            word.columns[HEAD] = point_to
                            root_to_define = None
                    else:
                        # if it is a list we point all of them to point_to
                        assert isinstance(root_to_define, list)
                        for root in root_to_define:
                            if word.columns[ID] == root:
                                print(f"HEURISTIC : writing : Word of index {word.columns[ID]} with head {word.columns[HEAD]} because {fix_mode} (list mode) set to root {point_to}")
                                word.columns[HEAD] = point_to
                                root_to_define_counter_fixed += 1
                        if root_to_define_counter_fixed == len(root_to_define):
                            root_to_define = None
                            fix_mode = "?"
                            root_to_define_counter_fixed = 0

                file.write("\t".join(word.columns)+"\n")
            file.write("\n")
            print(f"File written {writing}")

    class UDSpan:
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of span,
            # so we can use characters[start:end] or range(start, end).
            self.end = end

    class UDWord:
        def __init__(self, span, columns, is_multiword, deactivate_feat_processing=False):
            # Span of this word (or MWT, see below) within ud_representation.characters.
            self.span = span
            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word token.
            # In that case, self.span marks the span of the whole multi-word token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None if root).
            self.parent = None
            self.cycle = False

            # List of references to UDWord instances representing functional-deprel children.
            self.functional_children = []
            # Only consider universal FEATS.
            if not deactivate_feat_processing:
                self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                                  if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(":")[0]
            # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
            self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
            self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS

    ud = UDRepresentation()
    ud_fixed = UDRepresentation()
    # Load the CoNLL-U file
    index, sentence_start = 0, None
    cycle = 0
    cycle_for_span_start = []
    cycle_for_span_end = []
    cycle_ids = []

    while True:

        line = file.readline()
        if raise_all:
            pass#pdb.set_trace()
        if not line:
            break

        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            #pdb.set_trace()
            ud.sentences.append(UDSpan(index, 0))
            #sentence_start  is counted in number of words in the sentences
            sentence_start = len(ud.words)

        if not line:
            # Add parent and children UDWord links and check there are no cycles
            def process_word(word):
                #print("WORD.PARENT", word.columns, word.parent)
                if word.parent == "remapping":
                    try:
                        raise UDError(f"There is a cycle in a sentence word {word.columns} span {word.span.start} ")
                    except Exception as e:
                        if raise_all:
                            raise(Exception(e))
                        print(e)
                        word.cycle = True
                        #print()
                        cycle_ids.append(word.columns[0])
                        cycle_for_span_start.append(word.span.start)
                        cycle_for_span_end.append(word.span.end)

                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        try:
                            print("°°")
                            pdb.set_trace()
                            raise UDError("HEAD '{}' points outside of the sentence".format(word.columns))
                        except Exception as e:

                            raise(e)
                            #head = 1
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent
            #pdb.set_trace()
            # words have been loaded at these stage with their columns : now only checking and addign relation heads
            ind = 0
            len_cycle_for_span_start = len(cycle_for_span_start)
            for word in ud.words[sentence_start:]:
                process_word(word)
                # count root
            root_ind_ls = []
            for word in ud.words[sentence_start:]:
                if int(word.columns[HEAD]) == 0:
                    root_ind_ls.append(word.columns[ID])
                    ud_fixed.root_per_sent[sentence_start] = word.columns[ID]
            if len(root_ind_ls)>1:
                print("Several root",word.columns, root_ind_ls, sentence_start)
            if len(root_ind_ls) > 1 and not (len(cycle_for_span_start) > len_cycle_for_span_start):
                # no cycle but multiple root
                # take one

                if sentence_start not in ud_fixed.fixing_sents:
                    ud_fixed.fixing_sents[sentence_start] = OrderedDict()
                # we take all except last : last is the root that we keep in ud_fixed.root_per_sent[sentence_start]
                # NB : could be improve if need to pick more carefully
                ud_fixed.fixing_sents[sentence_start]["several_root_no_cycle"] = root_ind_ls[:-1]
            elif len(cycle_for_span_start) > len_cycle_for_span_start:
                # FIND THE CYCLES
                #_cycle_ids = OrderedDict()
                #for new_cycle in range(len(cycle_for_span_start)-len_cycle_for_span_start):

                for ind, word in enumerate(ud.words[sentence_start:]):
                    # cycle entry point

                    if str(ind + 1) == cycle_ids[-1-0]:
                        _cycle_ids = []
                        # find all the cycle
                        while True:
                            #if new_cycle not in _cycle_ids:
                            #    _cycle_ids[new_cycle] = []
                            _cycle_ids.append(word.columns[ID])
                            word = word.parent
                            if word.columns[ID] in _cycle_ids:
                                # we found all element of cycle : we break
                                print(f"We found all element of cycle of sent starts {sentence_start} i cycle in sent {0} len cycle {len(_cycle_ids)} {_cycle_ids}")
                                break

                #pdb.set_trace()
                #_cycle_ids = _cycle_ids[0][-1]
                assert len(_cycle_ids) > 0 #and len(_cycle_ids[0])>0
                #pdb.set_trace()
                if len(root_ind_ls) == 0:
                    # no root : pick one : you have a cycle so pick the root in a smart way
                    if sentence_start not in ud.fixing_sents:
                        ud_fixed.fixing_sents[sentence_start] = OrderedDict()
                    ud_fixed.fixing_sents[sentence_start]["no_root_and_cycle"] = _cycle_ids[0]
                elif len(root_ind_ls) == 1:
                    # one root : but cycle --> find the cycle and point to the root
                    if sentence_start not in ud.fixing_sents:
                        ud_fixed.fixing_sents[sentence_start] = OrderedDict()
                    ud_fixed.fixing_sents[sentence_start]["1_root_cycle"] = _cycle_ids[0]
                    pass
                elif len(root_ind_ls) > 1:
                    # several root and cycle --> pick 1 root and check if the cycle problem is still here
                    if sentence_start not in ud_fixed.fixing_sents:
                        ud_fixed.fixing_sents[sentence_start] = OrderedDict()
                        ud_fixed.fixing_sents[sentence_start]["several_root_and_cycle"] = OrderedDict()
                    ud_fixed.fixing_sents[sentence_start]["several_root_and_cycle"]["root"] = root_ind_ls[:-1]
                    ud_fixed.fixing_sents[sentence_start]["several_root_and_cycle"]["cycle"] = _cycle_ids[0]

                if word.cycle:
                    print("CYCLE", ind+sentence_start,  word.cycle)
                ind += 1
            #for word in ud.words[sentence_start:]:
                #if word.cycle:
                #    _word = deepcopy(word)
                #    #_word.columns[HEAD] = 0
                #    ud_fixed.words.append(_word)
                #else:
                #ud_fixed.words.append(word)
                    # fix it

                    # memor
            # func_children cannot be assigned within process_word
            # because it is called recursively and may result in adding one child twice.
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)

            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                try:
                    raise UDError(f"There are multiple roots in a sentence {word.columns}")
                except Exception as e:
                    if raise_all:
                        raise(Exception(e))
                    print(e)

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(line))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space.
        columns[FORM] = columns[FORM].replace(" ", "")
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(columns[ID]))
            ud_fixed.words.append(UDWord(ud.tokens[-1], columns, is_multiword=True, deactivate_feat_processing=True))
            #ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=True, deactivate_feat_processing=True))
            for _ in range(start, end + 1):
                _line = file.readline()
                word_line = _line.rstrip("\r\n")
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(word_line))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True, deactivate_feat_processing=False))
                ud_fixed.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True, deactivate_feat_processing=False))
                #UDWord(ud.tokens[-1], columns, is_multiword=True, deactivate_feat_processing=False)
                #if "-" in columns[0]:
                #pdb.set_trace()
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(columns[ID]))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(columns[ID], columns[FORM], len(ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(columns[HEAD]))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))
            ud_fixed.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud, ud_fixed


gold = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.4/fr_gsd-ud-test.conllu"
pred = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/parser/LAST_ep-prediction-fr_gsd-ud-test-.conll"


pred = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-prediction-fr_partut-ud-test-.conll"
gold = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-gold--fr_partut-ud-test-.conll"
fixed_file = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-pred_fix--fr_partut-ud-test-.conll"

pred = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-prediction-fr_spoken-ud-test-.conll"
gold = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-gold--fr_spoken-ud-test-.conll"

pred = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-prediction-fr_partut-ud-test-.conll"
gold = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-gold-fr_partut-ud-test-.conll"


fixed_file = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/LAST_ep-pred_fix--fr_spoken-ud-test-.conll"

fixed_file
raise_all = True
error = True
i_error = 0

file = open(pred, 'r')
while error:
    ud, ud_fix = load_conllu(file, False)
    pdb.set_trace()
    writing = open(fixed_file, 'w')
    ud_fix.write(writing)
    writing.close()
    print(f"{fixed_file} written")
    print("VALIDATION")
    #validated
    try:
        check_file = open(fixed_file, 'r')
        ud, ud_fix = load_conllu(check_file, raise_all=raise_all)
        error = False
        print(f"Done {fixed_file} fixed and validated after {i_error} pass")
    except Exception as e:
        i_error += 1
        print(f"Exception {i_error}, {e}")
        check_file.close()
        file = open(fixed_file, 'r')



dir_run_conll_eval = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/camembert/downstream/finetune/trainer/tools"
dir_run_conll_eval = " /Users/bemuller/Documents/Work/INRIA/dev/camembert/camembert/downstream/finetune/evaluate/"
dataset_name = "dev"
gold_file_name = gold
prediction_file = fixed_file
log_dir = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/models_demo/10136008-442dc-10136008_job-1a765_model-parsing-partut-ccnet-wwm-103k/"
print("Last checking")
os.system("cd {} && python {}/conll18_ud_eval-original.py  {} {}  ".format(log_dir, dir_run_conll_eval, prediction_file, prediction_file))
print("Eval checking")
os.system("cd {} && python {}/conll18_ud_eval-original.py  {} {} -vv ".format(log_dir, dir_run_conll_eval, gold_file_name, prediction_file))


# NOW EVAL
if "__main__" == __name__:

    file = open("/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.4/fr_gsd-ud-test.conllu")
    #load_conllu(file)

