import os
import json
import re
import string
import numpy as np
from torch import tensor
from torch._C import LoggerBase
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse

import numpy as np
import json
import pickle
from collections import defaultdict

from transformers.utils.dummy_pt_objects import load_tf_weights_in_funnel
from bart import MyBartModel
from span_utils import preprocess_span_input, preprocess_qpa, dump_pickle, load_pickle
import itertools
from numpy import random
from sklearn.cluster import KMeans
from dateutil.parser import ParserError, parse
from number_parser import parse_number


import csv




class QAData(object):

    def __init__(self, logger, args, data_path, dataset_type):
        """[summary]

        Args:
            logger ([type]): [description]
            args ([type]): [description]
            data_path ([type]): [description]
            dataset_type ([type]): ["train" or "dev"]

        Raises:
            NotImplementedError: [description]
        """
        self.data_path = data_path
        # determine is_training status now as dataset_type might be modfied later for file accessing
        self.is_training = dataset_type == "train"
        self.dataset_type = dataset_type

        if args.debug:
            self.data_path = data_path.replace("train", "dev")
            # under debug
            # we don't want to save train file as dev
            # we want to load dev file as train  (we simply don't save)
            dataset_type_for_file_accessing = "dev"
        else:
            if args.fine_tune:
                logger.info(
                    "Not AmbigQA test dataset available, using dev dataset")
                if not self.is_training:
                    dataset_type_for_file_accessing = "dev"  # fine tuning stage
                else:
                    dataset_type_for_file_accessing = dataset_type
            else:
                dataset_type_for_file_accessing = dataset_type
        # NOTE: self.data is the original data. Not tokenized nor encoded.
        with open(self.data_path, "r") as f:
            # format example: [ {'id': '-8178292525996414464', 'question': 'big little lies season 2 how many episodes', 'answer': ['seven']}, ..... ]
            self.data = json.load(f)
        if type(self.data) == dict:
            self.data = self.data["data"]
        self.processed_data = None
        if args.debug :
            if self.is_training == False:
                logger.warn("[DEBUG MODE] Load all dev data")
                self.data = self.data[:100]
            # else:
            #     self.data = self.data[100:]
            # logger.warn("[DEBUG MODE] Load partial dev data")
            # self.data = self.data[:500]

        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.index2id = {i: d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]: i for i, d in enumerate(self.data)}

        # TODO: correct it back
        self.load = True  # debug mode also needs load
        # self.load = not args.debug  # do not load the large tokenized dataset
        self.logger = logger
        self.args = args
        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()

        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.debug = args.debug
        self.answer_type = "span" if "extraction" in args.predict_type.lower(
        ) else "seq"  # TODO: condition on args.predict_type

        self.dataset_name = None  # ambig or nq
        self.passages = None
        if self.args.passage_clustering: # only need to load when using passage clustering
            self.clustered_passages_path = "data/clustering_results/AmbigQA_"
            postfix = ["top", self.args.top_k_passages, "passages",
                       self.data_type, "is_training", self.is_training, "rank_threshold", self.args.rank_threshold]
            postfix = [str(x) for x in postfix]
            postfix = "_".join(postfix)
            if self.args.debug:
                postfix += "_debug" # it might affect the number of data 
            self.clustered_passages_path  += postfix 
            # if os.path.exists(self.clustered_passages_path):
            #     with open(self.clustered_passages_path) as fp:
            #         self.passages = pickle.load(fp)
       
        self.question_ids = [d["id"] for d in self.data]
        
        

        # idea of naming detection is finding the folder name
        if any([n in args.ranking_folder_path for n in ["nq", "nqopen"]]):
            ranking_file_name = "nq_"
            data_file_n = "nqopen-"
            assert any(n in args.data_folder_path for n in ["nq", "nqopen"]) == True,\
                "data folder path/ranking_folder_path is wrong"
            assert any(n in self.data_path for n in ["nq", "nqopen"]) == True,\
                "data path/ranking_folder_path is wrong"
            self.dataset_name = "nq"
        elif any([n in args.ranking_folder_path for n in ["ambigqa"]]):
            ranking_file_name = "ambigqa_"
            data_file_n = "ambigqa_"  # NOTE: it's for light data only
            assert "ambigqa" in args.data_folder_path,\
                "data folder path/ranking_folder_path is wrong"
            assert "ambigqa" in self.data_path,\
                "data path/ranking_folder_path is wrong"
            self.dataset_name = "ambig"
        else:
            self.logger.warn("args.ranking_folder_path: ",
                             args.ranking_folder_path)
            exit()
        self.wiki_passage_path = args.passages_path
        self.ranking_path = os.path.join(
            args.ranking_folder_path, f"{ranking_file_name}{dataset_type_for_file_accessing}.json")
        self.data_path = os.path.join(
            args.data_folder_path, f"{data_file_n}{dataset_type_for_file_accessing}.json")
        self.top_k_passages = args.top_k_passages
        self.metric = "EM" if self.dataset_name == "nq" else "F1"
        self.sep_token = "<SEP>"
        self.spaced_sep_token = " " + self.sep_token + " "

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers, is_ambig=False):
        if not is_ambig:
            new_answers, metadata = [], []
            for answer in answers:
                metadata.append(
                    (len(new_answers), len(new_answers)+len(answer)))
                new_answers += answer

            return new_answers, metadata
        else:
            # sep token id
            new_answers, metadata = [], []
            # one data entry:   [ [singleQA["USA", "US"]], [multipleQA["CA", "Canada"], ["Mexico"]   ]   ]
            # _answers: []  answer for one data entry
            # answer:  answer for one annotation (singleQA or multipleQA)    [ [singleQA["USA", "US"]], [multipleQA["CA", "Canada"], ["Mexico"]   ]   ]
            # _answer: a list of acceptable answers for one
            
            for _answers in answers:
                assert type(_answers) == list
                metadata.append([])
                import pdb
                pdb.set_trace()
                # _answer: current: a list of acceptable answers:  [["US"], ["Canada"]]   expect: [["US", "USA"], ["Canada", "CA"]]
                for answer in _answers:
                    
                    metadata[-1].append([])
                    # current: "United States"    expect: ["United States", "USA"]
                    for _answer in answer:
                        
                        # one possibility: each singleAnswer qaPair has a list
                        assert len(_answer) > 0, _answers
                        assert type(_answer) == list and type(
                            _answer[0]) == str, _answers
                        # _answer should be a tuple of one answer

                        metadata[-1][-1].append((len(new_answers),
                                                 len(new_answers)+len(_answer)))
                        new_answers += _answer
            return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        logging_prefix = f"[{self.dataset_type} data]\t".upper()
        self.tokenizer = tokenizer

        # prepare paths and special token ids
        # NOTE:  Might have bug here

        # self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]
        # self.tokenizer.sep_token = self.sep_token # set tokenizer sep token make sure masking is working properly
        # For example: BartTokenizer -> BartTokenized
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        prepend_question_token = False
        if postfix[:2].lower() == "t5":  # TODO: find a smarter way to check if it's dataset for T5
           prepend_question_token = True
        if self.args.augment_k_times == 1:
            postfix = [postfix, "max_input_length", self.max_input_length, "top",  
                self.top_k_passages,  "rank_threshold", self.args.rank_threshold, self.answer_type, "is_training", self.is_training]  # TODO: can be written more elegantly by using dictionary
        else:
            postfix = [postfix, "max_input_length", self.max_input_length, "top",  
                self.top_k_passages, "rank_threshold", self.args.rank_threshold ,self.answer_type, "answers",  self.args.augment_k_times, "augmentation", "is_training", self.is_training]
        postfix = [str(x) for x in postfix]
        postfix = "_".join(postfix)
        if self.debug:
            postfix += "_debug"
        
        if self.args.passage_clustering:
            postfix += "_clustered"

        # TODO: decide to delete tokenized path if it's finally not needed
        tokenized_path = os.path.join(
            "/".join(self.data_path.split("/")[:-2]), "Tokenized",
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))  # replace .json by a formatted postfix
        clustered_passages_path = tokenized_path.replace(
            "Tokenized", "Clustered").replace(".json", "_input.p")
        wiki_embedding_path = "data/wiki2020embedding"
        encoded_input_path = tokenized_path.replace(
            "Tokenized", "Encoded").replace(".json", "_input.p")
        encoded_answer_path = tokenized_path.replace(
            "Tokenized", "Encoded").replace(".json", "_answer.p")
        metadata_path = tokenized_path.replace(
            "Tokenized", "Encoded").replace(".json", "_metadata.p")
        processed_data_path = encoded_input_path.replace("_input", "_data")

        def safe_remove(file_path):
            if os.path.exists(file_path):
                os.remove(file_path)
        def remove_confirmation_prompt(file_name):
            prompt = input(
                f"Confirm to remove {file_name}? (y/n)      ").lower()
            return  prompt == "yes"  or prompt == "y"
        if self.args.retokenize == True:
            if remove_confirmation_prompt("tokenization file"):
                safe_remove(tokenized_path)
            else:
                exit()


        if self.args.reencode == True:
            if remove_confirmation_prompt("encoding files"):
                safe_remove(processed_data_path)
                # safe_remove(encoded_answer_path)
                # safe_remove(metadata_path)
            else:
                exit()


        self.cache = os.path.exists(processed_data_path)

        joined_answers_l = []
        # load exists cache or pre-process a new one
        # General procedure:
        # 1. check if pickle cache exists
        # 2. if not, check if tokenized data exists
        # 3. if not, preprocess(load passages and encode) from scratch
        if self.load and self.cache:
            self.logger.info(
                logging_prefix + f"Found pickle cache, start loading {encoded_input_path}")
            if self.answer_type == "seq":
                # so we load encoding (it's batch + dictionary) and then pass then into

                # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                #     metadata, passage_coverage_rate = json.load(f)
                question_input, question_metadata, question_ids, answer_input, answer_metadata, joined_answers_l = load_pickle(
                    encoded_input_path)# , encoded_answer_path, metadata_path)

                self.question_ids = question_ids
                # import pdb; pdb.set_trace()

                input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
                decoder_input_ids, decoder_attention_mask = answer_input[
                    "input_ids"], answer_input["attention_mask"]
                if self.dataset_name == "ambig":
                    for (idx, joined_answers) in enumerate(joined_answers_l):
                        self.data[idx]["answers"] = joined_answers
                # inputs are lists of integers

            elif self.answer_type == "span":
                d = preprocess_span_input(
                    encoded_input_path, encoded_answer_path, metadata_path,
                    self.logger, tokenizer, self.max_input_length, is_training=self.is_training)
                input_ids = d["input_ids"]
                attention_mask = d["attention_mask"]
                token_type_ids = d["token_type_ids"]
                start_positions = d["start_positions"]
                end_positions = d["end_positions"]
                answer_mask = d["answer_mask"]
                # Q: input  (QA concatenation, y= answer?)
                # label is the start and end positions
                answer_coverage_rate = d["answer_coverage_rate"]
                self.logger.info(
                    logging_prefix + f"answer coverage rate by passages: {answer_coverage_rate}")

            else:
                self.logger.warn("wrong answer type")
                exit()
        else:  # not found pickle cache
            self.logger.info(logging_prefix +
                             "Not found pickle cache, start preprocessing...")
            

            if self.load and os.path.exists(tokenized_path): # found tokenized path
                self.logger.info(
                    logging_prefix + "Loading pre-tokenized data from {}".format(tokenized_path))
                with open(tokenized_path, "r") as f:
                    if self.answer_type == "seq":
                        input_ids, question_metadata, attention_mask, decoder_input_ids, decoder_attention_mask, \
                            answer_metadata, self.data, passage_coverage_rate = json.load(f)

                    elif self.answer_type == "span":
                        input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask, passage_coverage_rate = json.load(
                            f)

                    else:
                        self.logger.warn(logging_prefix +
                                         "Unrecognizable answer type")
                        exit()
                    self.logger.info(
                        logging_prefix + f"Passage kept rate(after truncation): {passage_coverage_rate * 100} %")
            else:  # not found tokenized data 
                self.logger.info(
                    logging_prefix + "Not found tokenized data, start tokenizing...")

                self.logger.info(
                    logging_prefix + "Not found tokenized data, start loading passagesing...")
                

                # pre-process question list from data
                questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                             for d in self.data]
                

                # NOTE: move code to data2


                # import pdb; pdb.set_trace()

                # pre-process answer list from data
                if self.dataset_name == "ambig":
                    answers = []
                    for (idx, data_entry) in enumerate(self.data):

                        cur_answer = []

                        # Q: does data_entry has more than one annotations? Or each answer is categorized
                        for qa_d in data_entry["annotations"]:
                            # import pdb
                            # pdb.set_trace()
                            if qa_d["type"] == "singleAnswer":
                                answer_for_one_qa_pair = [list(
                                    set(qa_d["answer"]))]  # a list of acceptable answers for one question interpretation

                                cur_answer.append(
                                    answer_for_one_qa_pair)
                            elif qa_d["type"] == "multipleQAs":
                                for pair in qa_d["qaPairs"]:
                                    answer_for_one_qa_pair = [list(
                                        set(pair["answer"]))]  # a list of semantic similar answers
                                    cur_answer.append(
                                        answer_for_one_qa_pair)
                            else:
                                self.logger.warn("error in qa_d type: ")
                                exit()
                        # cur_answer  [ [singleQA["USA", "US"]], [multipleQA["CA", "Canada"], ["Mexico"]   ]   ]
                        assert type(cur_answer) == list and \
                            all([type(answer) == list for answer in cur_answer]) and \
                            all([type(
                                _a) == str for answer in cur_answer for _answer in answer for _a in _answer])
                        answers.append(cur_answer)
                    # if self.answer_type == "span": # ambig span
                    #     answers = []
                    #     for (idx, d) in enumerate(self.data):
                    #         cur_answer = []
                    #         for qa_d in d["annotations"]:
                    #             if qa_d["type"] == "singleAnswer":
                    #                 # answers.append(qa_d["answer"])
                    #                 cur_answer.extend(qa_d["answer"])
                    #             elif qa_d["type"] == "multipleQAs":
                    #                 # answers.append(pair["answer"]) for pair in qa_d["qaPairs"]]
                    #                 pair_answers = []
                    #                 for pair in qa_d["qaPairs"]:
                    #                     pair_answers.extend(pair["answer"])
                    #                 cur_answer.extend(pair_answers)
                    #             else:
                    #                 self.logger.warn("error in qa_d type: ")
                    #                 exit()
                            
                    #         self.data[idx]["answers"] = cur_answer
                    #         # for one question, there is one list of answers
                    #         answers.append(cur_answer)
                    # elif self.answer_type == "seq": # ambig seq 
                    #     answers = []
                    #     for (idx, data_entry) in enumerate(self.data):

                    #         cur_answer = []

                    #         # Q: does data_entry has more than one annotations? Or each answer is categorized
                    #         for qa_d in data_entry["annotations"]:
                    #             import pdb; pdb.set_trace()
                    #             if qa_d["type"] == "singleAnswer":
                    #                 answer_for_one_qa_pair = [list(
                    #                     set(qa_d["answer"]))]  # a list of acceptable answers for one question interpretation

                    #                 cur_answer.append(
                    #                     answer_for_one_qa_pair)
                    #             elif qa_d["type"] == "multipleQAs":
                    #                 for pair in qa_d["qaPairs"]:
                    #                     answer_for_one_qa_pair = [list(
                    #                         set(pair["answer"]))]  # a list of semantic similar answers
                    #                     cur_answer.append(
                    #                         answer_for_one_qa_pair)
                    #             else:
                    #                 self.logger.warn("error in qa_d type: ")
                    #                 exit()
                    #         # cur_answer  [ [singleQA["USA", "US"]], [multipleQA["CA", "Canada"], ["Mexico"]   ]   ]
                    #         assert type(cur_answer) == list and \
                    #             all([type(answer) == list for answer in cur_answer]) and \
                    #             all([type(
                    #                 _a) == str for answer in cur_answer for _answer in answer for _a in _answer])
                    #         answers.append(cur_answer)
                    # else:
                    #     raise NotImplementedError()
                elif self.dataset_name == "nq":
                    answers = [d["answer"] for d in self.data]
                else:
                    self.logger.warn(
                        f"wrong dataset type: {self.dataset_name}")
                    exit()

                # flatten answer list
                answers, metadata = self.flatten(
                    answers, self.dataset_name == "ambig")

                if self.args.do_lowercase:
                    questions = [question.lower() for question in questions]
                    answers = [answer.lower() for answer in answers]

                # answers has been flattened, so it's normal to have more answers than questions

                self.logger.info(logging_prefix +
                                 "Start concatenating question and passages ")

                def init_top_k_passages():
                    if self.args.passage_clustering:
                        self.top_k_passages = 100
                        self.logger.info(
                            logging_prefix + "Passage clustering takes all (top 100) passages")
                        embedding_path = "data/wiki2020embedding/"
                        self.logger.info(
                            logging_prefix + "Loading passages embedding...")
                        passage_embedding = load_passage_embeddings(
                            embedding_path)
                        self.passages = topKPassasages(
                            self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path, passage_embedding=passage_embedding)

                    else:
                        self.passages = topKPassasages(
                            self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path)
                if self.answer_type == "seq":
                    if self.dataset_name == "nq":  # nq seq answer
                        questions = ["<s> " + q for q in questions]
                        # TODO: add them to arguments
                        # note that after this questions are actually a concatenation of questions and passages
                        self.logger.info(logging_prefix + f"Start concatenating question and passages for top {self.top_k_passages} passages")
                        self.passages = topKPassasages(
                            self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path)
                        for i in tqdm(range(len(questions))):
                            # mark the begining of passages
                            questions[i] += " <s> "
                            # add passage one by one
                            for p in self.passages.get_passages(i, self.args.top_k_passages):
                                # format: [CLS] question [SEP] title 1 [SEP] passages
                                questions[i] += self.spaced_sep_token + \
                                    p["title"] + self.spaced_sep_token + p["text"]
                            # mark the begining of passages
                            questions[i] += " </s> "
                        question_metadata = None 
                        answer_metadata = None
                        # NOTE: no need to rename
                        # questions_n_passages = questions  # rename
                        # new_questions = questions  # rename


                    elif self.dataset_name == "ambig":  # ambig seq answer
                        # TODO: add function pre_process in utils.py
                        if prepend_question_token:  # T5
                            questions = ["<s> question: " +
                                         question for question in questions]  # t5 tokenizer doesn't have <s>
                        else:
                            questions = ["<s> " + q for q in questions]  # Bart
                        questions = [q + " </s> " for q in questions]
                        questions_with_clustered_passages = []
                        # TODO: add them to arguments
                        # note that after this questions are actually a concatenation of questions and passages
                        all_qp_concatenation_list = []
                        self.logger.info(
                            logging_prefix + f"Start concatenating question and passages for top {self.top_k_passages} passages")
                        # import pdb; pdb.set_trace() 
                        num_clusters = 0
                        num_passages = 0
                        if self.args.passage_clustering and os.path.exists(self.clustered_passages_path): # check if there is clustered passagses (only need when passage clustering)
                            self.logger.info(
                                logging_prefix + "Loading clustering results...")
                            with open(self.clustered_passages_path,  "rb") as fp:
                                clustering_results = pickle.load(fp)
                                num_clusters = clustering_results["num_clusters"]
                                num_passages = clustering_results["num_passages"]
                                num_questions = clustering_results["num_questions"]
                                questions_n_passages = clustering_results["questions_n_passages"]
                        else: # no PC or PC but no clusteres passages

                            # load all passages embedding or 
                            
                            
                            init_top_k_passages() # init self.topKpassages

                            # if self.args.passage_clustering:
                            #     self.top_k_passages = 100
                            #     self.logger.info(
                            #         logging_prefix + "Passage clustering takes all (top 100) passages")
                            #     embedding_path = "data/wiki2020embedding/"
                            #     self.logger.info(
                            #         logging_prefix + "Loading passages embedding...")
                            #     passage_embedding = load_passage_embeddings(embedding_path)
                            #     self.passages = topKPassasages(
                            #         self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path, passage_embedding=passage_embedding)

                            # else:
                            #     self.passages = topKPassasages(
                            #         self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path)

                            import pdb
                            pdb.set_trace()
                            # concatenate question and passages
                            self.logger.info(
                                logging_prefix + "Concatenating questions and passages...")
                            preprocess_qpa(questions, self.passages, answers, 
                                            self.args.top_k_passages, 
                                            self.answer_type, self.is_ambig, self.args.passage_clustering, False, logging_prefix, self.logger)
                            if self.args.passage_clustering:
                                
                                for i in tqdm(range(len(questions))):
                                    clusters_passages, num_cluster_for_question_i, num_passages_for_question_i = self.passages.get_clustered_passages(
                                        i, self.args.rank_threshold)  # 2-d list
                                    num_clusters += num_cluster_for_question_i
                                    num_passages += num_passages_for_question_i 
                                    # make questions[i] a list, put index 0 a concatenation of all passsages
                                    all_qp_concatenation = questions[i]
                                    
                                    # add concatenation of all in the first entry
                                    for p_cluster in clusters_passages:
                                        all_qp_concatenation += " <s> "
                                        for p in p_cluster:
                                            # format: [CLS] question [SEP] title 1 [SEP] passages
                                            all_qp_concatenation += self.spaced_sep_token + \
                                                p["title"] + \
                                                self.spaced_sep_token + p["text"]
                                    all_qp_concatenation += " </s> "

                                    # cluster_qp_concatenation = questions[i]
                                    
                                    questions_with_clustered_passages.append([]) 
                                    questions_with_clustered_passages[i].append(all_qp_concatenation)
                                    
                                    
                                    for p_cluster in clusters_passages: # it's ordered 
                                        cluster_qp_concatenation = questions[i] # reset qp concatenation
                                        cluster_qp_concatenation += " <s> "
                                        start = True
                                        for p in p_cluster:
                                            # format: [CLS] question [SEP] title 1 [SEP] passages
                                            if start:
                                                cluster_qp_concatenation += p["title"] + \
                                                    self.spaced_sep_token + p["text"]
                                            else:
                                                cluster_qp_concatenation += self.spaced_sep_token + \
                                            p["title"] + self.spaced_sep_token + p["text"]
                                            start = False
                                        cluster_qp_concatenation += " </s> "
                                        questions_with_clustered_passages[i].append(
                                            cluster_qp_concatenation)
                                num_questions = len(questions) 
                                clustering_results = dict()
                                clustering_results["num_clusters"] = num_clusters
                                clustering_results["num_passages"] = num_passages
                                clustering_results["num_questions"] = num_questions
                                clustering_results["questions_n_passages"] = questions_with_clustered_passages
                                with open(self.clustered_passages_path, "wb") as fp:
                                    pickle.dump(clustering_results, fp)
                                # mark the begining of passages
                                questions_n_passages = questions_with_clustered_passages
                            else: # non-clustering
                                for i in tqdm(range(len(questions))):
                                    questions_n_passages = questions
                                    questions_n_passages[i] += " <s> " # start of passages
                                    # add passage one by one
                                    start = True
                                    # NOTE: get passage clustering
                                    for p in self.passages.get_passages(i, self.args.top_k_passages):
                                        # format: [CLS] question [SEP] title 1 [SEP] passages
                                        if start:
                                            questions_n_passages[i] += p["title"] + \
                                                self.spaced_sep_token + p["text"]
                                        else:
                                            questions_n_passages[i] += self.spaced_sep_token + \
                                                p["title"] + self.spaced_sep_token + p["text"]
                                        start = False
                                questions_n_passages[i] += " </s> "
                        if self.args.passage_clustering:
                            self.logger.info(
                                    f"Average number of clusters is (better be around 2): {num_clusters/len(questions)}")
                            self.logger.info(
                                f"Avg num of passages per cluster: {num_passages/num_clusters}")

                        

                        def is_answer_in_passages(answer_str, p_str):
                            """check the existance of answer in passages by comparing string

                            Args:
                                idx ([type]): [description]
                            """
                            return answer_str.lower() in p_str.lower()

                        def get_p_str(cur_qp, max_qp_length):
                            qp_ids = self.tokenizer.encode(
                                cur_qp)[:max_qp_length]
                            p_ids = qp_ids[qp_ids.index(eos_token_id):]
                            p_str = self.tokenizer.convert_ids_to_tokens(p_ids)
                            p_str = self.tokenizer.convert_tokens_to_string(
                                p_str)
                            return p_str
                        def concatenate_answers(answer_sets):
                            joined_answers = [answer for answer in itertools.product(*
                                                     answer_sets)]
                            concatenated_answers = [self.sep_token.join(
                                answer) for answer in joined_answers]
                            concatenated_answers = [
                                "<s>" + answer + "</s>" for answer in concatenated_answers]
                            # NOTE: add argument, num_k
                            max_num_of_answers = 100
                            if len(concatenated_answers) > max_num_of_answers:
                                rnd_indices = np.random.choice(
                                    len(concatenated_answers), size=max_num_of_answers, replace=False)
                                concatenated_answers = [concatenated_answers[i]
                                                        for i in rnd_indices]
                            return concatenated_answers

                        def is_answer_a_date_or_infreq_digit(answer_str):
                            # example: 2, 3, 4 -> False
                            # example:  july 25 2018 -> True
                            # example: more than 1 million -> it should be considered as a sequence of tokens which is more frequent 
                            try: 
                                if len(answer_str) < 4:  # # example: 2, 3, 4
                                    return False
                                parse(answer_str)
                                return True  # example:  july 25 2018
                            except ParserError:
                                answer_str = answer_str.replace(",","")
                                if parse_number(answer_str) != None:  # example: 55,000
                                    return True
                                return False  # example: more than 1 million
                            except TypeError:
                                return False  


                        new_questions = []
                        question_metadata = []
                        
                        new_answers = []
                        answer_metadata = []
                        question_indices = []
                        eos_token_id = self.tokenizer.eos_token_id
                        max_qp_length = self.args.max_input_length

                        # # new_questions
                        
                        # process answer for qp pair  (answer must be present in qp pair)
                        # Q, P, A must be processed at the same time because they are affecting each other in the training setting
                        num_eliminated_qp = 0
                        answer_presence_d = defaultdict(lambda: 0)
                        for idx, (cur_qp, cur_md) in enumerate(zip(questions_n_passages, metadata)):
                            if self.args.passage_clustering:
                                cur_qp_str = cur_qp[0] # in the past, we only check the whole concatenations
                            else:
                                cur_qp_str = cur_qp  # for non-PC, it should be normal top k passages 
                            p_str = get_p_str(cur_qp_str, max_qp_length)

                            # check existance of answers
                            found_answers_for_one_question = []

                            for cur_md_for_qa_pair in cur_md:
                                found_answer_for_qa_pair = []
                                for start, end in cur_md_for_qa_pair: # iterate acceptable answer (semantically similar answers)
                                    # acceptable answers for one qa pair
                                    answer_for_qa_pair = answers[start:end]
                                    for cur_a_str in answer_for_qa_pair:
                                        if self.is_training and not self.args.debug:
                                            if is_answer_in_passages(cur_a_str, p_str):
                                                found_answer_for_qa_pair.append(
                                                    cur_a_str)
                                        else:  # add all answers in eval dataset or any dataset in debug mode no matter its presence in passages 
                                            found_answer_for_qa_pair.append(
                                                cur_a_str)

                                if len(found_answer_for_qa_pair) > 0:
                                    found_answers_for_one_question.append(
                                        list(set(found_answer_for_qa_pair))) # NOTE: remove duplicated answers for one qa pair
                                
                            if len(found_answers_for_one_question) == 0 and self.is_training:
                                # actually in dev mode, length is certainly larger than zero as we will add answer no matter its presence in passages
                                continue
                            # NOTE: for regular training mode(no passage clustering), we still add answers for every question
                            if self.is_training and self.args.passage_clustering : # add answers separately for each clusters for training + passage clustering mode
                                # new type of 
                                # is_training -> seprate pairs of QP and A
                                # not is_training -> combine as we used to do
                                for (cluster_rank, cur_qp_str) in enumerate(cur_qp[1:]):
                                    aug_times = 0
                                    num_date = 0
                                    num_long_answer = 0
                                    found_answers_for_one_qp = []
                                    # check answer presence in all answers 
                                    # add presented answers into answer 
                                    for cur_md_for_qa_pair in cur_md:
                                        found_answer_for_qa_pair = []
                                        # iterate acceptable answer (semantically similar answers)
                                        for start, end in cur_md_for_qa_pair:
                                            # acceptable answers for one qa pair
                                            answer_for_qa_pair = answers[start:end]
                                            for cur_a_str in answer_for_qa_pair:
                                                # import pdb; pdb.set_trace()
                                                # print("cur_a_str: ", cur_a_str)
                                                # print("is_answer_in_passages: ", is_answer_in_passages(
                                                #     cur_a_str, cur_qp_str))
                                                if is_answer_in_passages(cur_a_str, cur_qp_str):
                                                    found_answer_for_qa_pair.append(
                                                        cur_a_str)
                                                    answer_presence_d[cluster_rank] += 1
                                                    if is_answer_a_date_or_infreq_digit(cur_a_str):
                                                        num_date += 1 
                                                    if len(cur_a_str.split(" ")) >= 4:
                                                        num_long_answer += 1 
                                        if len(found_answer_for_qa_pair) > 0:
                                            found_answers_for_one_qp.append(found_answer_for_qa_pair)
                                    # skip adding aligned questions and answers if not found qp 
                                    if len(found_answers_for_one_qp) == 0 and self.is_training:
                                        num_eliminated_qp += 1  # no actually eliminated because 
                                        empty_answer_gen_ratio = 0.5
                                        if np.random.rand() < empty_answer_gen_ratio:
                                            empty_answer_str = "<s> </s>"
                                            found_answer_for_qa_pair.append(
                                                empty_answer_str)
                                            
                                            found_answers_for_one_qp.append(
                                                found_answer_for_qa_pair)
                                        else:
                                            continue
                                    
                                    aug_times += len(found_answers_for_one_qp)
                                    aug_times += num_date
                                    aug_times += num_long_answer
                                    # concatenate qp's answers
                                    cur_answers = concatenate_answers(
                                        found_answers_for_one_qp)
                                    for i in range(aug_times):
                                        
                                        # append for each QP passages
                                        answer_start_idx = len(new_answers)
                                        # maintain its 1-D format
                                        new_answers.extend(cur_answers)
                                        answer_end_idx = len(new_answers)

                                        question_start_idx = len(new_questions)
                                        new_questions.append(cur_qp_str)
                                        question_end_idx = len(new_questions)

                                        question_metadata.append(
                                            (question_start_idx, question_end_idx)) # we actually added just a qp pair
                                        answer_metadata.append(
                                            (answer_start_idx, answer_end_idx))


                            else: # add concatenation of answers in eval dataset
                                joined_answers = [answer for answer in itertools.product(*
                                                                                         found_answers_for_one_question)]
                                joined_answers_l.append(joined_answers)
                                
                                concatenated_answers = [self.sep_token.join(
                                    answer) for answer in joined_answers]
                                concatenated_answers = [
                                    "<s>" + answer + "</s>" for answer in concatenated_answers]
                                # NOTE: add argument, num_k
                                max_num_of_answers = 100
                                if len(concatenated_answers) > max_num_of_answers:
                                    rnd_indices = np.random.choice(
                                        len(concatenated_answers), size=max_num_of_answers, replace=False)
                                    concatenated_answers = [concatenated_answers[i]
                                                            for i in rnd_indices]
                                cur_answers = concatenated_answers

                                answer_start_idx = len(new_answers)
                                # maintain its 1-D format
                                new_answers.extend(cur_answers)
                                answer_end_idx = len(new_answers)

                                question_start_idx = len(new_questions)

                                # rename for some clarity
                                question_id = self.question_ids[idx]
                                # import pdb; pdb.set_trace() 
                                if self.args.passage_clustering:
                                    # check cluster passages 
                                    new_questions.extend(cur_qp[1:])
                                    question_indices.extend(
                                        [question_id] * len(cur_qp[1:]))
                                else:
                                    new_questions.append(cur_qp_str)  # 
                                    question_indices.append(question_id)
                                # import pdb; pdb.set_trace()
                                # print("check first new_questions ")
                                question_end_idx = len(new_questions)
                                assert len(new_questions) == len(
                                    question_indices), "length shoudl be the same"
                                # TODO: find a way to save the question ids
                                
                                question_metadata.append(
                                    (question_start_idx, question_end_idx))
                                answer_metadata.append(
                                    (answer_start_idx, answer_end_idx))
                        
                        if self.args.passage_clustering:
                            import pdb; pdb.set_trace()
                            print("check answer_presence_d")
                            print("check num_eliminated_qp ")

                            self.logger.info(logging_prefix + f"Selected qp ratio: {len(question_metadata)/len(questions_n_passages)}")
                            self.logger.info(
                                logging_prefix + f"num_eliminated_qp")
                        
                        if len(question_indices) is not []:
                            self.question_ids = question_indices
                            # print("check question_ids set length")
                            # import pdb; pdb.set_trace()
                        questions = new_questions
                        answers = new_answers
                        # import pdb; pdb.set_trace()
                        print("answers example: ", answers[:30])
                        for (idx, joined_answers) in enumerate(joined_answers_l):
                            self.data[idx]["answers"] = joined_answers

                    
                    self.logger.info(
                        logging_prefix + f"Start encoding questions ({len(questions)}) and answers, this might take a while")
                    question_input = tokenizer.batch_encode_plus(questions,
                                                                 pad_to_max_length=True,
                                                                 max_length=self.args.max_input_length,
                                                                 truncation=True, 
                                                                 padding=True, 
                                                                 return_overflowing_tokens=True,
                                                                 verbose=self.args.verbose)

                    
                    max_answer_length = 30
                    answer_input = tokenizer.batch_encode_plus(answers,
                                                               pad_to_max_length=True,
                                                               max_length=max_answer_length,
                                                               truncation=True,
                                                               padding=True, 
                                                               verbose=self.args.verbose)
                    dump_pickle(question_input, question_metadata, self.question_ids, answer_input, answer_metadata, joined_answers_l, encoded_input_path,
                                )

                    input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
                    decoder_input_ids, decoder_attention_mask = answer_input[
                        "input_ids"], answer_input["attention_mask"]
                    # if not self.is_training:
                    #     decoder_input_ids=  None
                    #     decoder_attention_mask = None
                    #     metadata = None
                    num_truncated_tokens = abs(sum(
                        question_input['num_truncated_tokens']))     
                    num_quesiton_ids = sum(
                        [len(question) for question in question_input['input_ids']])
                    passage_coverage_rate = num_quesiton_ids / \
                        (num_truncated_tokens + num_quesiton_ids)
                    self.logger.info(
                        logging_prefix + f"Number of truncated tokens: {num_truncated_tokens}")
                    self.logger.info(
                        logging_prefix + f"Passage kept rate(after truncation): {passage_coverage_rate * 100} %")
                elif self.answer_type == "span":
                    question_metadata = None
                    # assume questions = [Q1, Q2]
                    # answers = [[A1 <SEP> A2], [A3]]
                    # all titles = [ [T1, T2, ..., T100], [T1, T2, ..., T100]   ]
                    # TODO: add some of these arguments into questions
                    all_titles = []
                    all_passages = []
                    init_top_k_passages()
                    # for each question, add a list of passages info from reranking results
                    # all titles and all passages should be a 2-d list
                    for i in tqdm(range(len(questions))):
                        cur_titles = []
                        cur_passages = []

                        for p in self.passages.get_passages(i, self.args.top_k_passages):
                            cur_titles.append(p["title"])
                            cur_passages.append(p["text"])
                        all_titles.append(cur_titles)
                        all_passages.append(cur_passages)
                    self.logger.info(logging_prefix +
                                     "Start preprocessing span input")
                    d = preprocess_span_input(
                        encoded_input_path, encoded_answer_path, metadata_path,
                        self.logger, tokenizer, self.max_input_length,
                        questions=questions, answers=answers, metadata=metadata, all_titles=all_titles, all_passages=all_passages, is_training=self.is_training)

                    input_ids = d["input_ids"]
                    attention_mask = d["attention_mask"]
                    token_type_ids = d["token_type_ids"]
                    start_positions = d["start_positions"]
                    end_positions = d["end_positions"]
                    answer_mask = d["answer_mask"]
                    # Q: input  (QA concatenation, y= answer?)
                    # label is the start and end positions
                    answer_coverage_rate = d["answer_coverage_rate"]

                else:
                    print("Unrecognizable answer type")
                    exit()
                if self.load:
                    with open(tokenized_path, "w") as fp:
                        if self.answer_type == "seq":
                            json.dump([input_ids, question_metadata, attention_mask,
                                       decoder_input_ids, decoder_attention_mask,
                                       answer_metadata, self.data, passage_coverage_rate], fp)
                        elif self.answer_type == "span":
                            json.dump([input_ids, attention_mask, token_type_ids, start_positions,
                                       end_positions, answer_mask, answer_coverage_rate], fp)
        
        # loading dataset
        if self.answer_type == "seq":
            question_metadata = None  # as I shift to use question_indices instead
            self.dataset = QAGenDataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        passage_clustering=self.args.passage_clustering,
                                        question_ids=self.question_ids,
                                        in_metadata=question_metadata, out_metadata=answer_metadata,
                                        is_training=self.is_training)
        elif self.answer_type == "span":
            # batch size x max_n_answer
            list_of_tensors = self.tensorize(
                input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask)
            self.dataset = TensorDataset(*list_of_tensors)
        else:
            print("wrong self.answer_type argument")
            exit()
        self.logger.info(
            logging_prefix + "Loaded {} examples from {} data".format(len(self.dataset), self.data_type))
        # make sure all questions are included in evaluation mode

        # it no longer work for clustered passages 
        # if not self.is_training:
            # assert len(input_ids) == len(self), (len(input_ids), len(self))
        self.logger.info("DEV length check has passed")
        if do_return:
            return self.dataset

    def tensorize(self, *args):
        """Transform list of tensors into a tensor with uniform size

        Args:
            l ([type]): [description]
        """
        list_of_tensors = []
        for l in args:
            max_tensor_len = max([len(t) for t in l])
            new_tensor = torch.zeros(len(l), max_tensor_len,  dtype=torch.long)
            for i in range(len(l)):
                t = l[i]
                if type(t) == list:
                    t = torch.LongTensor(t)
                new_tensor[i, : t.size(0)] = t
            list_of_tensors.append(new_tensor)
        return list_of_tensors

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(
            self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        """Evaluate exact matches

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        if type(predictions) is not defaultdict and type(predictions[0]) == list: # check defaultdict first to avoid adding empty list
            # each answer is a list of all acceptable answers.   [str, str]
            self.answer_type = "span"
        else:
            # each answer is concatenation of all accpetable answers.   str
            self.answer_type = "seq"
        # import pdb; pdb.set_trace()
        assert len(predictions) == len(self), (len(predictions), len(self))
        ems = []
        f1s = []

        # TODO
        if self.dataset_name == "ambig":
            if self.answer_type == "seq":
                for (prediction, dp) in zip(predictions, self.data):
                    if type(predictions) == defaultdict:
                        question_idx = prediction # it's actually dictionary key, predictions is a dictionary here
                        prediction = predictions[question_idx]
                    cur_answers = dp["answers"] 
                    # for qa_d in dp["annotations"]:
                    #     if qa_d["type"] == "singleAnswer":
                    #         cur_answer.extend(qa_d["answer"])
                    #     elif qa_d["type"] == "multipleQAs":
                    #         pair_answers = []
                    #         for pair in qa_d["qaPairs"]:
                    #             pair_answers.extend(pair["answer"])
                    #         cur_answer.extend(pair_answers)
                    #     else:
                    #         self.logger.warn("error in qa_d type: ")
                    #         exit()
                   
                    # import pdb; pdb.set_trace()

                    # regular f1
                    # prediction = prediction.replace(
                        # "<s>", "").replace("</s>", "").split("<sep>")
                    # max_f1 = np.max( [get_f1(cur_answer, prediction)  for cur_answer in cur_answers])

                    # f1 with out duplication
                    prediction=prediction.replace(
                        "<s>", "").replace("</s>", "").split("<sep>")
                    prediction = [normalize_answer(pred) for pred in prediction] 
                    prediction = [pred for pred in prediction if len(pred) != 0 ] # remove empty prediction



                    max_f1 = np.max([get_f1(list(set(cur_answer)), list(set(prediction)))
                                     for cur_answer in cur_answers] ) 



                    print(f"f1: {max_f1}  prediction: {prediction} cur_answer: {cur_answers[:10]}")
                    # NOTE: the only difference from span answer type
                    f1s.append(max_f1)
            else:
                for (prediction, dp) in zip(predictions, self.data):
                    cur_answer =[]
                    for qa_d in dp["annotations"]:
                        if qa_d["type"] == "singleAnswer":
                            # answers.append(qa_d["answer"])
                            cur_answer.extend(qa_d["answer"])
                        elif qa_d["type"] == "multipleQAs":
                            # answers.append(pair["answer"]) for pair in qa_d["qaPairs"]]
                            pair_answers = []
                            for pair in qa_d["qaPairs"]:
                                pair_answers.extend(pair["answer"])
                            cur_answer.extend(pair_answers)
                        else:
                            self.logger.warn("error in qa_d type: ")
                            exit()
                    f1s.append(get_f1(cur_answer, prediction))
            return f1s
        elif self.dataset_name == "nq":
            if self.answer_type == "seq":
                for (prediction, dp) in zip(predictions, self.data):
                    # there are many concatenation of answers and they are all correct
                    # we append the one with the highest score
                    
                    ems.append(get_exact_match(prediction, dp["answer"]))
            else:
                for (prediction, dp) in zip(predictions, self.data):
                    for pred in prediction:
                        ems.append(get_exact_match(pred, dp["answer"]))
            return ems
        # def get_exact_match(prediction, groundtruth):
        # if type(groundtruth)==list:
        #     if len(groundtruth)==0:
        #         return 0
        #     return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
        # return (normalize_answer(prediction) == normalize_answer(groundtruth)

    def save_predictions(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.prefix}predictions_top_{self.args.top_k_answers}_answers.json")
        
        if self.args.passage_clustering:
            # For PC predictions, it's a dictionary already have question ids as keys
            if type(predictions) == dict:
                with open(save_path, "w") as f:
                    json.dump(predictions, f)
        else:
            prediction_dict = {dp["id"]: prediction for dp,
                            prediction in zip(self.data, predictions)}
            with open(save_path, "w") as f:
                json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_exact_match(prediction, groundtruth):
    if type(groundtruth) == list:  # for ambigQA answer input
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


def get_f1(answers, predictions, is_equal=get_exact_match):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    if len(predictions) == 0:
        return 0 
    assert len(answers) > 0


    # assert len(answers) > 0 and len(predictions) > 0, (answers, predictions)

    
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers) == np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if a+b == 0:
        return 0
    return 2*a*b/(a+b)


class QAGenDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask, 
                 passage_clustering = False, 
                 question_ids = None, 
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training
        self.passage_clustering = passage_clustering
        self.question_ids = question_ids
        if self.passage_clustering and is_training == False:
            assert question_ids is not None, "Must have question ids in PC model eval mode"
        assert len(self.input_ids) == len(
            self.attention_mask) == self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids) == len(
            self.decoder_attention_mask) == self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            # if self.passage_clustering:
            #     indices = self.in_metadata[idx]
            #     # import pdb; pdb.set_trace()
            #     # print("expect idx to be a range of indices of correct answers")
            #     # return a list of QP and attention mask
            #     input_ids_list = [self.input_ids[idx] for idx in range(*indices)]
            #     attention_mask = [self.attention_mask[idx]
            #                       for idx in range(*indices)]
            #     assert type(
            #         input_ids_list) == list, "input_ids_list should be 2 d: " + str(input_ids_list)
            #     # normalized_indices = 
            #     return input_ids_list, attention_mask, indices

            #     # return the similar data as the last 
            # else:
            #     idx = self.in_metadata[idx][0]
            #     return self.input_ids[idx], self.attention_mask[idx]
            idx = self.in_metadata[idx][0]
            try:
                return self.input_ids[idx], self.attention_mask[idx], self.question_ids[idx]
            except IndexError:
                import pdb; pdb.set_trace()

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


def load_passage_embed(i, f_name, embedding_path):
    with open(embedding_path + f'{f_name}_{i}.pkl', 'rb') as f:
        return  (i, pickle.load(f))  # (key, value) pair


def load_passage_embeddings(embedding_path):
    embedding_data = []  # embedding can be accessed by simply using passage id

    # # NOTE: for debugging purpose, here we only load 20 passage embedding files
    # for i in tqdm(range(50)):
    #     with open(embedding_path + f'wiki2020embedding_{i}.pkl', 'rb') as f:
    #         embedding_data.extend(pickle.load(f))

    import multiprocessing as mp
    pool = mp.Pool(20)
    # import pdb; pdb.set_trace()
    if "2020" in embedding_path:
        f_name = "wiki2020embedding"
    else:
        f_name = "wikipedia_passages"
    
    embedding_d = dict(pool.starmap_async(load_passage_embed, zip(range(50),[f_name]*50 , [embedding_path]*50 )  ).get()   )
    pool.close()
    pool.join()

    for i in tqdm(range(50)):
        embedding_data.extend(embedding_d[i])
    # import pdb; pdb.set_trace()
    # print("check if there is way to reduce RAM usage")
    return embedding_data

class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training and not args.debug: 
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            # sampler = RandomSampler(dataset)
            # in debug mode, we can see how non-identical data help training
            sampler = SequentialSampler(dataset)
            # TODO: add forcing change predict batch size
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(
            dataset, sampler=sampler, batch_size=batch_size)


class topKPassasages():
    """
    This class serves as a modular way of retrieving top k passages of a question for reader
    """

    def __init__(self, k, passages_path, rank_path, data_path, passage_embedding = None, evaluate=False):
        # load wiki passages and store in dictionary

        # a list of lists of passsages ids   [ [ 3,5, ], ...  ]
        self.ranks = self.load_ranks(rank_path)
        self.answers = self.load_answer(data_path)
        # for nq dataset, {id:str, question:text, answer:text}
        # for ambig dataset, {id:str, question:text, answer:[text1, text2]} ?
        # a list of dictionary {title:str, text:str}

        wiki_split_path = passages_path.split("/")
        wiki_split_path[-1] = wiki_split_path[-1].replace(".tsv", "_split")
        wiki_split_path = "/".join(wiki_split_path)

        import os
        # split wiki passages 
        # check if passage split data is available
        if not os.path.exists(wiki_split_path) or len(os.listdir(wiki_split_path)) == 0:
            self.passages = self.load_passages(
                passages_path, wiki_split_path, parallel = False)  # a list of dictionary

        else:
            print("loading passages in parallel  ")
            self.passages = self.load_passages(
                passages_path, wiki_split_path, parallel=True)
    
        self.passage_embeddings = passage_embedding
        
        if evaluate:
            # self.recall = self.evaluate_recall()
            self.evaluate_macro_avg_recall()
        # only keep top k passages during initialization

        print("end of top k passages class")


    
    def get_clustered_passages(self, i, rank_threshold):
        """Indexed on quesiton id and return clusters of passages

        Args:
            i ([type]): [description]
        Returns:
            [type]: [description]
        """
        
        passage_embeddings = self.get_passage_embeddings(
            i)
        kmeans_1 = KMeans(n_clusters=10, random_state=0).fit(passage_embeddings)
        # from IPython import embed; embed()


        # compute stat of clusters
        cluster_pts_count = dict()
        for j in range(10):
            cluster_pts_count[j] = sum(
                kmeans_1.labels_ == j)
        # for j in range(10):
        #     print(j, ": ", sum(kmeans_1.labels_ == j))
        cluster_ranks = dict()
        

        # add up ranks
        for j in range(0, len(kmeans_1.labels_)):
            cluster_label = kmeans_1.labels_[j]
            # TODO: defaultdict
            if cluster_label in cluster_ranks.keys():
                cluster_ranks[cluster_label] += j
            else:
                cluster_ranks[cluster_label] = j 
        # average ranks
        for j in range(10):
            cluster_ranks[j] /= cluster_pts_count[j]
        
        sorted_cluster_ranks=  sorted(cluster_ranks.items(),
                key=lambda item: item[1])
        # we want the smallest few. (ranked higher)
        print(sorted_cluster_ranks)



        # add top-k cluster
        filtered_clusters = []
        for (cluster_label, avg_rank) in sorted_cluster_ranks:
            if avg_rank < rank_threshold:
                filtered_clusters.append(
                    cluster_label)
            else:
                break
        if len(filtered_clusters) == 0:
            filtered_clusters.append(sorted_cluster_ranks[0][0]) # append the first cluster label 


        passages = []
        passage_ids = self.ranks[i]

        num_cluster_passages_l = []
        # add top 5 passages' ids from each cluster 
        for cluster_label in filtered_clusters:
        
            cluster_passages = []
            for j in range(len(kmeans_1.labels_)):  # iterate all cluster labels and keep the order

                if kmeans_1.labels_[j] == cluster_label:
                    passage_id = passage_ids[j]

                    cluster_passages.append(
                        self.passages[passage_id])
                if len(cluster_passages) == 5: # TODO: change 5 to top_k argument
                    break
            num_cluster_passages_l.append(len(cluster_passages) )
            assert len(cluster_passages) > 0 and len(cluster_passages) <= 5, "each cluster should have more than one passages and less than five passages"
            passages.append(cluster_passages)
        assert len(passages) >= 1, "There should be more than one cluster"
        num_cluster_for_question_i = len(num_cluster_passages_l)
        num_selected_cluster_passages_for_question_i = sum(num_cluster_passages_l)
        
        return passages, num_cluster_for_question_i, num_selected_cluster_passages_for_question_i


    def get_passages(self, i, k):
        """
        0-indexed based retrieval to get top k passages.
        Note that rank, answers and passages are lists with the same length
        :param i: index
        :return: a list of passage dictionary {title:str, text:str}
        """
        top_k_ranked_passage_ids = self.ranks[i][:k]
        # get rank prediction
        return [self.passages[passage_id] for passage_id in top_k_ranked_passage_ids]
    
    def get_passage_embeddings(self, i):
        assert self.passage_embeddings is not None, "passage embedding is not loaded"
        passage_embeddings = []
        
        for passage_id in self.ranks[i]:
            try:
                passage_embeddings.append(self.passage_embeddings[passage_id])
            except IndexError:
                print("index error")
                continue
        passage_embeddings = [embed[1] for embed in passage_embeddings]
        return passage_embeddings
        # return [self.passage_embeddings[passage_id][1] for passage_id in self.ranks[i]]

    def get_passage_ids(self, i):
        return self.ranks[i]

    def load_wiki_data(self, passages_path):
        wiki_data = []
        with open(passages_path, "rb") as fp:
            for line in fp.readlines():

                wiki_data.append(line.decode().strip().split("\t"))

                    
        return wiki_data

    def load_wiki_split(self, passages_path, split_idx ):
        wiki_data = []
        with open(passages_path, "r") as fp:
            for line in fp.readlines():
                wiki_data.append(line.strip().split("\t"))
        return (split_idx, wiki_data)
        

    def write_wiki_data(self, split_idx, wiki_split_path, f_name,  num_p_per_split, wiki_data):
        with open(os.path.join(wiki_split_path, f_name + f"_{split_idx}.tsv"), 'w') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            for i in range(split_idx*num_p_per_split, min((split_idx+1)*num_p_per_split, len(wiki_data))):
                tsv_output.writerow(wiki_data[i])


    def load_passages(self, passages_path, wiki_split_path, parallel = False) -> list:
        """[load, format passages ]

        Args:
            passages_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        f_name = wiki_split_path.split("/")[-1]
        num_split = 30  # set to the number of threads on CPU
        # wiki_data = []
        import multiprocessing as mp
        # original passage file is binary file
        # passage split is not binary file
        if not parallel:
            if not os.path.exists(wiki_split_path):
                os.makedirs(wiki_split_path)
            wiki_data_split = []
            wiki_data = self.load_wiki_data(passages_path)
            # with open(passages_path, "rb") as fp:
            #     for line in fp.readlines():
            #         import pdb; pdb.set_trace()
            #         wiki_data.append(line.decode().strip().split("\t"))
            
            num_p_per_split = len(wiki_data) // num_split
            assert wiki_data[0] == ["id", "text", "title"]

            # save split files 
            import csv

            # pool = mp.Pool()
            # pool.starmap(self.write_wiki_data, zip(
            #     range(num_split), [wiki_split_path]*num_split, [f_name]*num_split, [num_p_per_split]*num_split))
            for split_idx in range(num_split):
                self.write_wiki_data(
                    split_idx, wiki_split_path, f_name,  num_p_per_split, wiki_data)

            # TODO: don't we record passage id? id is just its index (we change it to 0 based)
        else:
            # load split files directly
            

            # import pdb; pdb.set_trace() 
            # print("check RAM usage before loading passages")
            pool = mp.Pool(30)
            wiki_data = []
            # use dictionary to presever the order
            passage_split_paths = [os.path.join(wiki_split_path, f_name + f"_{split_idx}.tsv") for split_idx in range(num_split)]
            wiki_data_d = dict(pool.starmap_async(self.load_wiki_split, zip(
                passage_split_paths,  range(num_split))  ).get())
            pool.close()
            pool.join()
            for i in range(num_split):
                wiki_data.extend(wiki_data_d[i])
        # transform
        wiki_data = [{"title": title, "text": text}
                     for _, text, title in wiki_data[1:]]
        return wiki_data

    def load_answer(self, data_path):
        # load answer for the question
        with open(data_path, "r") as fp:
            answers = json.load(fp)
        return answers

    def load_ranks(self, rank_path):
        # load
        with open(rank_path, "r") as fp:
            ranks = json.load(fp)  # 0-indexed ranks
        return ranks

    def evaluate_macro_avg_recall(self):
        """evalute annotation recall
        """
        top_k_passages_recall = defaultdict(
            list)  # keep track of top k passages maximum recall

        for d, passage_indices in zip(self.answers, self.ranks):
            assert len(passage_indices) == 100
            answers = []  # collect answers for the annotations
            # collecting answers based on the annotation type
            for qa_d in d["annotations"]:
                if qa_d["type"] == "singleAnswer":
                    # answers.append(qa_d["answer"])
                    answers.append([qa_d["answer"]])
                elif qa_d["type"] == "multipleQAs":
                    # answers.append(pair["answer"]) for pair in qa_d["qaPairs"]]
                    answers.append([pair["answer"]
                                    for pair in qa_d["qaPairs"]])
                else:
                    print("error in qa_d type")
            passages = [normalize_answer(
                self.passages[passage_index]["text"]) for passage_index in passage_indices]
            for k in [1, 5, 10, 100]:
                answers_recall = []
                passages_str = " ".join(passages[:k])
                # answer = [ [sometime one answer string], [sometimes a list of acceptable strings]  ]
                # For example, [["Canada"], ["USA", "United States", "United States of America"]]
                for answer in answers:
                    token_presence = [int(any([normalize_answer(
                        _answer_token) in passages_str for _answer_token in answer_token])) for answer_token in answer]
                    cur_recall = sum(token_presence)/len(token_presence)
                    answers_recall.append(cur_recall)
                top_k_passages_recall[k].append(max(answers_recall))

        for k in [1, 5, 10, 100]:
            print("Recall @ %d\t%.1f%%" %
                  (k, 100*np.mean(top_k_passages_recall[k])))
        return top_k_passages_recall

    def evaluate_recall(self):
        recall = defaultdict(list)
        for d, passage_indices in zip(self.answers, self.ranks):
            assert len(passage_indices) == 100
            answers = [normalize_answer(answer) for answer in d["answer"]]
            passages = [normalize_answer(
                self.passages[passage_index]["text"]) for passage_index in passage_indices]
            for k in [1, 5, 10, 100]:
                recall[k].append(
                    any([answer in passage for answer in answers for passage in passages[:k]]))

        for k in [1, 5, 10, 100]:
            print("Recall @ %d\t%.1f%%" % (k, 100*np.mean(recall[k])))
        return recall
