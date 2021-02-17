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
from span_utils import preprocess
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
        # TODO: correct it back
        self.load = not args.debug  # do not load the large tokenized dataset
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
        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.concatenateQA = False
        self.debug = args.debug
        self.answer_type = "span" # TODO: condition on args.predict_type

        # provides paths for pasting convenience
        
        #  data/ambigqa/ambigqa_train.json


        # load passages dataset
        # TODO: add tokenized path
        # TODO: add data naming and folder architecture into readme
        if args.predict_type == "SpanSeqGen":
            self.concatenateQA = True

            self.k = args.top_k
            wiki_passage_path = args.passages_path

            # idea of naming detection is finding the folder name 
            if any([ n in args.ranking_folder_path for n in ["nq", "nqopen"]]):
                ranking_file_n = "nq_"
                data_file_n = "nqopen-"
            elif  any([n in args.ranking_folder_path for n in ["ambigqa"]]):
                ranking_file_n = "ambigqa_"
                data_file_n = "ambigqa_" # NOTE: it's for light data only 
            else: 
                pass
            ranking_path = os.path.join(args.ranking_folder_path, f"{ranking_file_n}{dataset_type}.json")
            data_path = os.path.join(args.data_folder_path, f"{data_file_n}{dataset_type}.json")
            self.passages = topKPassasages(self.k, wiki_passage_path, ranking_path, data_path)
        elif args.predict_type == "SpanExtraction":
            self.concatenateQA = True

            self.k = args.top_k
            wiki_passage_path = args.passages_path
            ranking_path = os.path.join(args.ranking_folder_path, f"{ranking_file_n}{dataset_type}.json")
            data_path = os.path.join(args.data_folder_path, f"{data_file_n}{dataset_type}.json")
            self.passages = topKPassasages(self.k, wiki_passage_path, ranking_path, args.data_path) 
        else:
            print("Wrong predict_type!")
            exit()

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
        answer_type = "span"  # TODO: add it to argument
        self.tokenizer = tokenizer
        tokenizer.add_tokens(["<SEP>"]) # add extra token for BART 
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")  # For example: BartTokenizer -> BartTokenized
        prepend_question_token = False
        if postfix[:2].lower() == "t5": # TODO: find a smarter way to check if it's dataset for T5
           prepend_question_token = True
        postfix = "_".join([postfix, "max_input_length", str(self.max_input_length), "top",  str(self.k), answer_type]) # TODO: can be written more elegantly by using dictionary
        if self.debug:
            postfix += "_debug"
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        

        if self.load and os.path.exists(preprocessed_path): 
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                if answer_type == "gen": 
                    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                        metadata, passage_coverage_rate = json.load(f)
                    print("Passage coverage rate: ", passage_coverage_rate * 100, " %")
                elif answer_type == "span":
                    input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask = json.load(
                        f)
                else:
                    print("Unrecognizable answer type")
                    exit()   
        else:
            print ("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            if self.debug:
                questions = questions[:100]
                answers = answers[:100]
            answers, metadata = self.flatten(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]



            if answer_type == "gen":
                if prepend_question_token:
                       questions = ["question: " + question for question in questions] 
                questions = ["<s> " + q for q in questions]
                # TODO: add them to arguments
                # note that after this questions are actually a concatenation of questions and passages
                print("Start concatenating question and passages for top ", self.k , " passages")
                for i in tqdm(range(len(questions))):
                    for p in self.passages.get_passages(i): # add passage one by one
                        questions[i] += " <SEP> " + p["title"] + " <SEP> " + p["text"] # format: [CLS] question [SEP] title 1 [SEP] passages
                    questions[i] += " </s>"

                print("Encoding questions and answers, this might take a while")
                question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length,
                                                         return_overflowing_tokens=True)
                answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True)
            # add answer type variable 
            # two types of question answering
                input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
                decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
                
                num_truncated_tokens = sum(question_input['num_truncated_tokens']) 
                num_quesiton_ids = sum(  [ len(question) for question in  question_input['input_ids'] ] ) 
                passage_coverage_rate =   num_quesiton_ids    / (num_truncated_tokens + num_quesiton_ids) 
                print("Passage coverage rate: ", passage_coverage_rate * 100, " %")
            elif answer_type == "span":
                # assume questions = [Q1, Q2]
                # answers = [[A1 <SEP> A2], [A3]]
                # all titles = [ [T1, T2, ..., T100], [T1, T2, ..., T100]   ]
                # TODO: add some of these arguments into questions 
                all_titles = []
                all_passages = []
                is_training = self.data_type == "train" 
                max_n_answers = 5

                for i in tqdm(range(len(questions))):
                    cur_titles = []
                    cur_passages = []
                    for p in self.passages.get_passages(i):
                        cur_titles.append(p["title"])
                        cur_passages.append(p["text"])
                    all_titles.append(cur_titles)
                    all_passages.append(cur_passages)
                # NOTE: pdb 
                # import pdb
                # pdb.set_trace()
                d = preprocess(tokenizer, questions, answers, metadata, all_titles, all_passages, \
                    is_training, self.max_input_length, max_n_answers)
                input_ids = d["input_ids"]
                attention_mask = d["attention_mask"] 
                token_type_ids  = d["token_type_ids"]
                start_positions = d["start_positions"] 
                end_positions = d["end_positions"]
                answer_mask = d["answer_mask"]
                # Q: input  (QA concatenation, y= answer?)
                # label is the start and end positions





            else:
                print("Unrecognizable answer type")
                exit()     
            if self.load:
                # save tokenized data in training mode
                # if answer_type == "gen":
                #     preprocessed_data = [input_ids, attention_mask,
                #                         decoder_input_ids, decoder_attention_mask,
                #                         metadata, passage_coverage_rate]
                    
                    
                # elif answer_type == "span":
                #     import pdb
                #     pdb.set_trace()
                #     preprocessed_data = d
                #     # self.dataset = QASpanDataset(*d.values()) 
                #     # self.dataset = Dataset(input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask)
                #     list_of_tensors = self.tensorize(
                #         input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask)
                #     self.dataset = TensorDataset(*list_of_tensors)

                with open(preprocessed_path, "w") as fp:
                    if answer_type == "gen":
                        json.dump([input_ids, attention_mask,
                                decoder_input_ids, decoder_attention_mask,
                                metadata, passage_coverage_rate], fp)
                    elif answer_type == "span":
                         json.dump([input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask], fp) 
            if answer_type == "gen":
                self.dataset = QAGenDataset(input_ids, attention_mask,
                                            decoder_input_ids, decoder_attention_mask,
                                            in_metadata=None, out_metadata=metadata,
                                            is_training=self.is_training)
            elif answer_type == "span":
                # import pdb
                # pdb.set_trace()

                # self.dataset = QASpanDataset(*d.values())
                # self.dataset = Dataset(input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask)
                list_of_tensors = self.tensorize(
                    input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask)
                import pdb
                pdb.set_trace()
                self.dataset = TensorDataset(*list_of_tensors)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

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
            # import pdb
            # pdb.set_trace()
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



class QASpanDataset(Dataset):
    # Q: do I need this class of dataset? 
    # Q: if so, do I need negative dataset? I don't think so as it's was used to train dpr retriever
    def __init__(self, input_ids, attention_mask, token_type_ids, start_positions, end_positions, answer_mask) -> None:
        """[summary]
        Tensorize list input

        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
            start_positions ([type]): [description]
            end_positions ([type]): [description]
            answer_mask ([type]): [description]
            offset_mapping ([type]): [description]
            raw_inputs ([type]): [description]
        """

        import pdb
        pdb.set_trace()

        # M is number of passage per question
        # input ids = number of question x number of passages
        self.input_ids = self.tensorize(input_ids)   # 40 x non-uniform list
        self.attention_mask = self.tensorize(attention_mask) 
        self.token_type_ids = self.tensorize(token_type_ids) 
        self.start_positions = self.tensorize(start_positions) 
        self.end_positions = self.tensorize(end_positions) 
        self.answer_mask = self.tensorize(answer_mask) 
       
    def tensorize(self, data):
        return [torch.LongTensor(d) for d in data] 

    def _pad(self, input_ids, M):
        # input_ids is a tensor of one input
        # if no input ids, then return zeros tensor
        if len(input_ids)==0:
            return torch.zeros((M, self.negative_input_ids[0].size(1)), dtype=torch.long)
        # stack input ids
        if type(input_ids)==list:
            input_ids = torch.stack(input_ids)
        # 
        if len(input_ids)==M:
            return input_ids
        return torch.cat([input_ids,
                            torch.zeros((M-input_ids.size(0), input_ids.size(1)), dtype=torch.long)],
                            dim=0)

    def __len__(self):
        return len(self.in_metadata)
    
    def __getitem__(self, idx):
        """There are two types of data
        Input data: question, attention_mask
        Output data:  answer

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        # in eval mode, return data by order
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            input_ids = self.positive_input_ids[idx][:self.test_M]
            attention_mask = self.positive_input_mask[idx][:self.test_M]
            token_type_ids = self.positive_token_type_ids[idx][:self.test_M] # it shows what are valid tokens
            # return self.input_ids[idx], self.attention_mask[idx]
            return [self._pad(t, self.test_M) for t in [input_ids, attention_mask, token_type_ids]]
        # randomly return data in training mode
        # NOTE: check in_metadata and out_metadata
        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))  # where the answer is 

        return self.input_ids[in_idx], self.attention_mask[in_idx], self.token_type_ids[in_idx], \
            self.start_positions[out_idx], self.end_positions[out_idx], self.answer_mask[out_idx]

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
        

        # check answers 
        import pdb
        pdb.set_trace()
        if evaluate:
            self.recall = self.evaluate_recall()
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


    def evaluate_macro_avg_recall(self):
        """evalute annotation recall

        """
        import pdb
        pdb.set_trace()

        top_k_passages_recall = defaultdict(list)# keep track of top k passages maximum recall 


        # compute maximum recall for each passages
        
        
        for d, passage_indices in zip(self.answers, self.ranks):
            assert len(passage_indices)==100
            answers = [] # collect answers for the annotations
            # collecting answers based on the annotation type
            for qa_d in d["annotations"]:
                if qa_d["type"] == "singleAnswer":
                    answers.append([qa_d["answer"]])
                elif qa_d["type"] == "multipleQAs":
                    answers.append([pair["answer"] for pair in qa_d["qaPairs"]])
                else:
                    print("error in qa_d type")
            passages = [normalize_answer(self.passages[passage_index]["text"]) for passage_index in passage_indices] 
            print(passages[0]) 
            # convert the list of answers to a list of recall scores
            # take the maximum
            for k in [1,5,10,100]:
                answers_recall = []
                passages_str = " ".join(passages[:k])
                
                print(passages_str)
                # print("answers: ", answers)
                for answer in answers:
                    # each answer should be a list of list of strings
                    # print("answer: ", answer)
                    token_presence = [int(any([normalize_answer(_answer_token) in passages_str for _answer_token in answer_token])) for answer_token in answer ]
                    cur_recall = sum(token_presence)/len(token_presence)
                    answers_recall.append(cur_recall)
                # print(type(k))
                # print("answers_recall", answers_recall)
                # print("max answers recall", max(answers_recall))
                top_k_passages_recall[k].append(max(answers_recall))

        for k in [1,5,10,100]:
            print ("Recall @ %d\t%.1f%%" % (k, 100*np.mean(top_k_passages_recall[k]))) 
        return top_k_passages_recall




        for k in [1,5,10,100]:
            print("Recall @ %d\t%.1f%%" % (k, 100*np.mean(recall[k])))

           # (Multiple QA) qaPairs -> iterate qa pair -> answer
            # (single answer) iterate a list with length one -> answer

        # average out the recall 
        pass 

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

    def evaluate_macro_average_recall(self):
        pass  
