import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse

import numpy as np
import json
import data
import pickle
from collections import defaultdict
from bart import MyBartModel
from span_utils import preprocess_span_input, dump_pickle, load_pickle
import itertools
from numpy import random
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
        self.is_training = dataset_type == "train"  # determine is_training status now as dataset_type might be modfied later for file accessing
        self.dataset_type =dataset_type 

        if args.debug:
            self.data_path = data_path.replace("train", "dev")
            dataset_type_for_file_accessing = "dev"
        else:
            if args.fine_tune:
                logger.info("Not AmbigQA test dataset available, using dev dataset")
                if not self.is_training:
                    dataset_type_for_file_accessing = "dev"  # fine tuning stage
                else:
                    dataset_type_for_file_accessing = dataset_type # 
            else:
                dataset_type_for_file_accessing = dataset_type 
        # NOTE: self.data is the original data. Not tokenized nor encoded.
        with open(self.data_path, "r") as f:
            self.data = json.load(f)  # format example: [ {'id': '-8178292525996414464', 'question': 'big little lies season 2 how many episodes', 'answer': ['seven']}, ..... ]
        if type(self.data)==dict:
            self.data = self.data["data"]
        if args.debug and self.is_training == False:
            logger.warn("[DEBUG MODE] Load all dev data")
            self.data = self.data[:]
            # logger.warn("[DEBUG MODE] Load partial dev data")
            # self.data = self.data[:500]

        assert type(self.data)==list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"])==int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])
        # import pdb; pdb.set_trace()


        self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]:i for i, d in enumerate(self.data)}

        

        # TODO: correct it back
        self.load = True  # debug mode also needs load 
        # self.load = not args.debug  # do not load the large tokenized dataset
        # self.load =  args.debug
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
        self.answer_type = "span" if "extraction" in args.predict_type.lower() else "seq" # TODO: condition on args.predict_type
        
        self.dataset_name = None # ambig or nq
        self.passages = None
        
        # idea of naming detection is finding the folder name
        if any([n in args.ranking_folder_path for n in ["nq", "nqopen"]]):
            ranking_file_name = "nq_"
            data_file_n = "nqopen-"
            assert any(n in args.data_folder_path for n in ["nq", "nqopen"] ) == True,\
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
            self.logger.warn("args.ranking_folder_path: ", args.ranking_folder_path)
            exit()
        self.wiki_passage_path = args.passages_path
        self.ranking_path = os.path.join(
            args.ranking_folder_path, f"{ranking_file_name}{dataset_type_for_file_accessing}.json")
        self.data_path = os.path.join(
            args.data_folder_path, f"{data_file_n}{dataset_type_for_file_accessing}.json")
        self.top_k_passages = args.top_k_passages
        self.metric = "EM" if self.dataset_name == "nq" else "F1"
        

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
                metadata.append((len(new_answers), len(new_answers)+len(answer)))
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
                for answer in _answers: # _answer: current: a list of acceptable answers:  ["US", "Canada"]   expect: [["US", "USA"], ["Canada", "CA"]]
                    metadata[-1].append([])
                    for _answer in answer:  # current: "United States"    expect: ["United States", "USA"]
                        # one possibility: each singleAnswer qaPair has a list 
                        import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        # NOTE:  Might have bug here 
        if self.args.model == "bart":
            self.tokenizer.add_tokens(["<SEP>"]) # add extra token for BART 
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")  # For example: BartTokenizer -> BartTokenized
        prepend_question_token = False
        if postfix[:2].lower() == "t5": # TODO: find a smarter way to check if it's dataset for T5
           prepend_question_token = True
        if self.args.augment_k_times == 1:
            postfix = "_".join([postfix, "max_input_length", str(self.max_input_length), "top",  str(
                self.top_k_passages), self.answer_type])  # TODO: can be written more elegantly by using dictionary
        else:
            postfix = "_".join([postfix, "max_input_length", str(self.max_input_length), "top",  str(
                self.top_k_passages), self.answer_type, "answers",  self.args.augment_k_times, "augmentation"])
        if self.debug:
            postfix += "_debug"
        # TODO: decide to delete tokenized path if it's finally not needed
        tokenized_path = os.path.join(
            "/".join(self.data_path.split("/")[:-2]), "Tokenized",
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))  # replace .json by a formatted postfix
        encoded_input_path = tokenized_path.replace("Tokenized", "Encoded").replace(".json", "_input.p")
        encoded_answer_path = tokenized_path.replace("Tokenized", "Encoded").replace(".json", "_answer.p")
        metadata_path = tokenized_path.replace(
            "Tokenized", "Encoded").replace(".json", "_metadata.p")
        # 1. check if there is cache, if not then tokenize. If there is cache, we 
        # 2. check if
        # import pdb; pdb.set_trace(); print("check encoded path") 
        self.cache = os.path.exists(encoded_input_path) \
                and os.path.exists(encoded_answer_path) \
                and os.path.exists(metadata_path)

        # General procedure:
        # 1. check if pickle cache exists
        # 2. if not, check if tokenized data exists
        # 3. if not, preprocess(load passages and encode) from scratch

        if self.load and self.cache:
            self.logger.info(logging_prefix + f"Found pickle cache, start loading {encoded_input_path}")
            if self.answer_type == "seq":
                # so we load encoding (it's batch + dictionary) and then pass then into 

                # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                #     metadata, passage_coverage_rate = json.load(f)
                question_input, answer_input, metadata = load_pickle(
                    encoded_input_path, encoded_answer_path, metadata_path)
                input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
                decoder_input_ids, decoder_attention_mask = answer_input[
                    "input_ids"], answer_input["attention_mask"]
                # inputs are lists of integers 
                

            elif self.answer_type == "span":
                d = preprocess_span_input(
                    encoded_input_path, encoded_answer_path, metadata_path, \
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
                self.logger.info(logging_prefix + f"answer coverage rate by passages: {answer_coverage_rate}")
                 
            else:
                self.logger.warn("wrong answer type")
                exit()
        else: 
            self.logger.info(logging_prefix + "Not found pickle cache, start preprocessing...")
            
            if self.load and os.path.exists(tokenized_path): 
                self.logger.info(logging_prefix + "Loading pre-tokenized data from {}".format(tokenized_path))
                with open(tokenized_path, "r") as f:
                    if self.answer_type == "seq": 
                        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                            metadata, passage_coverage_rate = json.load(f)
                        
                    elif self.answer_type == "span":
                        input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask, passage_coverage_rate = json.load(
                            f)
                        
                    else:
                        self.logger.warn(logging_prefix + "Unrecognizable answer type")
                        exit()   
                    self.logger.info(logging_prefix + f"Passage coverage rate: {passage_coverage_rate * 100} %")
            else:
                self.logger.info(logging_prefix + "Not found tokenized data, start tokenizing...")

                self.logger.info(logging_prefix + "Not found tokenized data, start loading passagesing...")
                self.passages = topKPassasages(
                    self.top_k_passages, self.wiki_passage_path, self.ranking_path, self.data_path)
                
                self.args.augment_k_times
                questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                            for d in self.data]
                # NOTE: this code is untested yet
                if self.dataset_name == "ambig":
                    if self.answer_type == "span":
                        answers = []
                        for d in self.data:
                            cur_answer = []
                            for qa_d in d["annotations"]:
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
                            
                            answers.append(cur_answer) # for one question, there is one list of answers
                    elif self.answer_type == "seq":
                        answers = []
                        num_of_permutations_l = []
                        num_of_tokens_l = []
                        for data_entry in self.data:

                            cur_answer = []
                            
                            # Q: does data_entry has more than one annotations? Or each answer is categorized 
                            for qa_d in data_entry["annotations"]:
                                if qa_d["type"] == "singleAnswer":
                                    cur_answer.append(  [list(set(qa_d["answer"]))] )
                                    # answers.append(qa_d["answer"])
                                    # if len(cur_answer) == 0:
                                    #     cur_answer = qa_d["answer"]
                                    # else: # cases cur_answer is already a list of acceptable concatenated answers
                                    #     cur_answer = itertools.product(
                                    #         cur_answer, qa_d["answer"])
                                        
                                        
                                elif qa_d["type"] == "multipleQAs":
                                    # cur_answer = itertools.product(
                                    #     *cur_answer)
                                    for pair in qa_d["qaPairs"]:
                                        cur_answer.append([list(set(pair["answer"]))])
                                else:
                                    self.logger.warn("error in qa_d type: ")
                                    exit()
                            # cur_answer  [ [singleQA["USA", "US"]], [multipleQA["CA", "Canada"], ["Mexico"]   ]   ]
                            assert type(cur_answer) == list and \
                                all([type(answer) == list for answer in cur_answer]) and \
                                all([
                                    type(_a) == str for answer in cur_answer for _answer in answer for _a in _answer])

                            # how to slice answer to ensure tokens lower than 
                            # cur_answer = [answer[:num_answers_kept] for answer in cur_answer]  # [(1,2,3,....36), (1,2,3,...,37)] -> [(1,2,3,...,10), (1,2,3,...,10)]
                            # cur_answer = itertools.product(*cur_answer)
                            # cur_answer = [item for item in cur_answer]
                            # if len(cur_answer) > 30:
                            #     rnd_indices = np.random.choice(
                            #         len(cur_answer), size=30, replace=False)
                            #     cur_answer = [cur_answer[i]
                            #                     for i in rnd_indices]



                            # print("num of permutations: ",
                            #         len(cur_answer))
                            # num_of_permutations_l.append(
                            #     len(cur_answer))
                            # num_of_tokens_l.append(
                            #     len(cur_answer[0]))
                            # it's better to transform it later after flatening
                            # cur_answer = [" <SEP> ".join(answer) for answer in cur_answer]
                            # cur_answer = [answer + " </s>" for answer in cur_answer]
                            

                            # NOTE: the following code is correct
                            # answers is like the level of self.data 
                            answers.append(cur_answer) 



                            # check answer coverage
                            
                            # 1. number of tokens. (limit to 30 and check its coverage)
                            # 2. number of permutations and how many to keep. (sampling strategy, )
                            # 3. oversampling multi-answer QA 
                            # 4. padding


                            # 1. augumentation depending on the number of questions 
                            # 2. cut out answer based on len(token_id) not len(cur_anwer)
                            # 3. How the answers are kept in data get_item which causes the OOM?
                            # type id and mask 


                            # answers 1. one qa 2. annotation 3. str
                                


                        
                        
                    else:
                        raise NotImplementedError()
                elif self.dataset_name == "nq":
                    answers = [d["answer"] for d in self.data]
                else:
                    self.logger.warn(
                        f"wrong dataset type: {self.dataset_name}")
                    exit()
                
                answers, metadata = self.flatten(answers, self.dataset_name == "ambig")
                
                if self.args.do_lowercase:
                    questions = [question.lower() for question in questions]
                    answers = [answer.lower() for answer in answers]

                import pdb;pdb.set_trace()

                print("check length of questions and answers, and add post-processing")

                self.logger.info(logging_prefix +  "Start concatenating question and encoding")
                if self.answer_type == "seq":

                    # TODO: add function pre_process in utils.py  
                    if prepend_question_token:
                        questions = ["question: " + question for question in questions] 
                    questions = ["<s> " + q for q in questions]
                    # TODO: add them to arguments
                    # note that after this questions are actually a concatenation of questions and passages
                    print(logging_prefix + "Start concatenating question and passages for top ", self.top_k_passages , " passages")
                    for i in tqdm(range(len(questions))):
                        for p in self.passages.get_passages(i): # add passage one by one
                            questions[i] += " <SEP> " + p["title"] + " <SEP> " + p["text"] # format: [CLS] question [SEP] title 1 [SEP] passages
                        questions[i] += " </s>"

        
                    self.logger.info(
                        logging_prefix + "Start encoding questions and answers, this might take a while")
                    question_input = tokenizer.batch_encode_plus(questions,
                                                            pad_to_max_length=True,
                                                            max_length=self.args.max_input_length,
                                                            return_overflowing_tokens=True,
                                                            verbose=self.args.verbose)
                    answer_input = tokenizer.batch_encode_plus(answers,
                                                               pad_to_max_length=True, verbose=self.args.verbose)
                    dump_pickle(question_input, answer_input, metadata, encoded_input_path,
                                 encoded_answer_path, metadata_path)
                    # add answer type variable 
                    # two types of question answering
                    input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
                    decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
                    
                    num_truncated_tokens = sum(question_input['num_truncated_tokens']) 
                    num_quesiton_ids = sum(  [ len(question) for question in  question_input['input_ids'] ] ) 
                    passage_coverage_rate =   num_quesiton_ids    / (num_truncated_tokens + num_quesiton_ids) 
                    self.logger.info(
                        logging_prefix + f"Passage coverage rate: {passage_coverage_rate * 100} %")
                elif self.answer_type == "span":
                    # assume questions = [Q1, Q2]
                    # answers = [[A1 <SEP> A2], [A3]]
                    # all titles = [ [T1, T2, ..., T100], [T1, T2, ..., T100]   ]
                    # TODO: add some of these arguments into questions 
                    all_titles = []
                    all_passages = []
                    

                    # for each question, add a list of passages info from reranking results 
                    # all titles and all passages should be a 2-d list
                    for i in tqdm(range(len(questions))):
                        cur_titles = []
                        cur_passages = []
                        
                        for p in self.passages.get_passages(i):
                            cur_titles.append(p["title"])
                            cur_passages.append(p["text"])
                        all_titles.append(cur_titles)
                        all_passages.append(cur_passages)
                    self.logger.info(logging_prefix +
                                     "Start preprocessing span input")
                    d = preprocess_span_input(
                        encoded_input_path, encoded_answer_path, metadata_path, \
                        self.logger, tokenizer, self.max_input_length,
                        questions=questions, answers=answers, metadata=metadata, all_titles=all_titles, all_passages=all_passages, is_training=self.is_training)

                    input_ids = d["input_ids"]
                    attention_mask = d["attention_mask"] 
                    token_type_ids  = d["token_type_ids"]
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
                            json.dump([input_ids, attention_mask,
                                    decoder_input_ids, decoder_attention_mask,
                                    metadata, passage_coverage_rate], fp)
                        elif self.answer_type == "span":
                            json.dump([input_ids, attention_mask, token_type_ids, start_positions,
                                    end_positions, answer_mask, answer_coverage_rate], fp)
        
        # loading dataset
        if self.answer_type == "seq":
            self.dataset = QAGenDataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        in_metadata=None, out_metadata=metadata,
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
        if not self.is_training:
            assert len(input_ids) == len(self), ( len(input_ids), len(self))
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
                new_tensor[i, : t.size(0)] =  t
            list_of_tensors.append(new_tensor)
        return list_of_tensors

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader




    def evaluate(self, predictions):
        """Evaluate exact matches

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        if type(predictions[0])== list:
            self.answer_type = "span" # each answer is a list of all acceptable answers.   [str, str]
        else:
            self.answer_type = "seq"  # each answer is concatenation of all accpetable answers.   str
        
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        f1s = []
        # from IPython import embed; embed()

        # TODO 
        if self.dataset_name == "ambig":
            if self.answer_type == "seq": 
                raise NotImplementedError()
            else:
                for (prediction, dp) in zip(predictions, self.data):
                    cur_answer = []
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
        elif self.dataset_name ==  "nq":
            if self.answer_type == "seq":
                for (prediction, dp) in zip(predictions, self.data):
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
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.prefix}predictions_top_{self.args.top_k_answers}_answers.json")
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
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))



def get_f1(answers, predictions, is_equal=get_exact_match):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
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
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if a+b==0:
        return 0
    return 2*a*b/(a+b)

# def get_exact_match(prediction, groundtruth):
#     if type(groundtruth)==list:
#         if len(groundtruth)==0:
#             return 0
#         return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
#     return (normalize_answer(prediction) == normalize_answer(groundtruth))




class QAGenDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
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

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            # sampler = RandomSampler(dataset)
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)



class topKPassasages():
    """
    This class serves as modular way of retrieving top k passages of a question for reader
    """
    def __init__(self, k, passages_path, rank_path, data_path, evaluate=False):
        # load wiki passages and store in dictionary

        
        self.ranks = self.load_ranks(rank_path) # a list of lists of passsages ids   [ [ 3,5,9 ], ...  ]
        self.answers = self.load_answer(data_path)
        # for nq dataset, {id:str, question:text, answer:text}
        # for ambig dataset, {id:str, question:text, answer:[text1, text2]} ?
        self.passages = self.load_passages(passages_path) # a list of dictionary {title:str, text:str}
        

        if evaluate:
            # self.recall = self.evaluate_recall()
            self.evaluate_macro_avg_recall()
        self.topKRank(k)

    def get_passages(self, i):
        """
        0-indexed based retrieval to get top k passages.
        Note that rank, answers and passages are lists with the same length
        :param i: index
        :return: a list of passage dictionary {title:str, text:str}
        """
        # get rank prediction
        try: 
            return [self.passages[passage_id] for passage_id in self.ranks[i]]
        except IndexError:
            import pdb
            pdb.set_trace()


    def topKRank(self, k=10):
        self.ranks = [r[:k] for r in self.ranks]

    def load_passages(self, passages_path):
        """[load, format passages ]

        Args:
            passages_path ([type]): [description]

        Returns:
            [type]: [description]
        """

        wiki_data = []
        with open(passages_path, "rb") as fp:
            for line in fp.readlines():
                wiki_data.append(line.decode().strip().split("\t"))
        assert wiki_data[0]==["id", "text", "title"]
        wiki_data = [ {"title": title, "text": text} for _, text, title in wiki_data[1:]]  # TODO: don't we record passage id? id is just its index (we change it to 0 based)
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
        top_k_passages_recall = defaultdict(list)# keep track of top k passages maximum recall 
        
        for d, passage_indices in zip(self.answers, self.ranks):
            assert len(passage_indices)==100
            answers = [] # collect answers for the annotations
            # collecting answers based on the annotation type
            for qa_d in d["annotations"]:
                if qa_d["type"] == "singleAnswer":
                    # answers.append(qa_d["answer"])
                    answers.append([qa_d["answer"]])
                elif qa_d["type"] == "multipleQAs":
                    # answers.append(pair["answer"]) for pair in qa_d["qaPairs"]]
                    answers.append([pair["answer"] for pair in qa_d["qaPairs"]])
                else:
                    print("error in qa_d type")
            passages = [normalize_answer(self.passages[passage_index]["text"]) for passage_index in passage_indices] 
            for k in [1,5,10,100]:
                answers_recall = []
                passages_str = " ".join(passages[:k])
                # answer = [ [sometime one answer string], [sometimes a list of acceptable strings]  ]
                # For example, [["Canada"], ["USA", "United States", "United States of America"]]
                for answer in answers:
                    token_presence = [int(any([normalize_answer(_answer_token) in passages_str for _answer_token in answer_token])) for answer_token in answer ] 
                    cur_recall = sum(token_presence)/len(token_presence)
                    answers_recall.append(cur_recall)
                # print("answers_recall", answers_recall)
                # print("max answers recall", max(answers_recall))
                top_k_passages_recall[k].append(max(answers_recall))

        for k in [1,5,10,100]:
            print ("Recall @ %d\t%.1f%%" % (k, 100*np.mean(top_k_passages_recall[k]))) 
        return top_k_passages_recall

    def evaluate_recall(self):
        recall = defaultdict(list)
        for d, passage_indices in zip(self.answers, self.ranks):
            assert len(passage_indices)==100
            answers = [normalize_answer(answer) for answer in d["answer"]]  
            passages = [normalize_answer(self.passages[passage_index]["text"]) for passage_index in passage_indices]
            for k in [1, 5, 10, 100]:
                recall[k].append(any([answer in passage for answer in answers for passage in passages[:k]]))

        for k in [1, 5, 10, 100]:
            print ("Recall @ %d\t%.1f%%" % (k, 100*np.mean(recall[k])))
        return recall
