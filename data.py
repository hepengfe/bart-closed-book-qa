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
from tqdm import tqdm

class QAData(object):

    def __init__(self, logger, args, data_path, dataset_type):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        with open(self.data_path, "r") as f:
            self.data = json.load(f)  # format example: [ {'id': '-8178292525996414464', 'question': 'big little lies season 2 how many episodes', 'answer': ['seven']}, ..... ]
        if type(self.data)==dict:
            self.data = self.data["data"]
        if args.debug:
            self.data = self.data[:40]
        assert type(self.data)==list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"])==int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])
                 


        self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]:i for i, d in enumerate(self.data)}
        if dataset_type == "train":
            self.is_training = True
        else:
            self.is_training = False
        self.load = not args.debug
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
        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.concatenateQA = False


        # load passages dataset
        if args.predict_type == "SpanSeqGen":
            self.concatenateQA = True

            self.k = args.top_k
            wiki_passage_path = args.passages_path
            ranking_path = os.path.join(args.ranking_path, f"nq_{dataset_type}.json")
            data_path = os.path.join(args.data_path, f"nqopen-{dataset_type}.json")
            self.passages = topKPassasages(self.k, wiki_passage_path, ranking_path, data_path)

                

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
            
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        tokenizer.add_tokens(["<SEP>"]) # add extra token for BART 
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")  # For example: BartTokenizer -> BartTokenized
        postfix = "_".join([postfix, "max_input_length", str(self.max_input_length), "top",  str(self.k)]) # TODO: can be written more elegantly by using dictionary
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        if self.load and os.path.exists(preprocessed_path): 
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata, passage_coverage_rate = json.load(f)
                print("Passage coverage rate: ", passage_coverage_rate * 100, " %")
        else:
            print ("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.concatenateQA:
                questions = ["<s> " + q for q in questions]

                # TODO: add them to arguments
                # note that after this questions are actually a concatenation of questions and passages
                print("Start concatenating question and passages for top ", self.k , " passages")
                for i in tqdm(range(len(questions))):
                    for p in self.passages.get_passages(i): # add passage one by one
                        questions[i] += " <SEP> " + p["title"] + " <SEP> " + p["text"] # format: [CLS] question [SEP] title 1 [SEP] passages
                    questions[i] += " </s>"
            else:
                append_qa_token = True
                if append_qa_token:
                    # questions = ["question: "+question for question in questions]
                    questions = ["question: "+question for question in questions]
                if self.args.append_another_bos:
                    questions = ["<s> "+question for question in questions]
                    answers = ["<s> " +answer for answer in answers]






            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length,
                                                         return_overflowing_tokens=True)
            
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True)

            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            
            num_truncated_tokens = sum(question_input['num_truncated_tokens']) 
            num_quesiton_ids = sum(  [ len(question) for question in  question_input['input_ids'] ] ) 
            passage_coverage_rate =   num_quesiton_ids    / (num_truncated_tokens + num_quesiton_ids) 
            print("Passage coverage rate: ", passage_coverage_rate * 100, " %")
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata, passage_coverage_rate]
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata, passage_coverage_rate], f)
        self.dataset = MyQADataset(input_ids, attention_mask,
                                         decoder_input_ids, decoder_attention_mask,
                                         in_metadata=None, out_metadata=metadata,
                                         is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        for (prediction, dp) in zip(predictions, self.data):
            ems.append(get_exact_match(prediction, dp["answer"]))
        return ems

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

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


class MyQADataset(Dataset):
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
        self.answers = self.load_answer(data_path) # {id:str, question:text, answer:text}
        self.passages = self.load_passages(passages_path) # a list of dictionary {title:str, text:str}

        if evaluate:
            self.recall = self.evaluate_recall()


        self.topKRank(k)

    def get_passages(self, i):
        """
        0-indexed based retrieval to get top k passages.
        Note that rank, answers and passages are lists with the same length
        :param i: index
        :return: a list of passage dictionary {title:str, text:str}
        """
        # get rank prediction
        return [self.passages[passage_id] for passage_id in self.ranks[i]]


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

