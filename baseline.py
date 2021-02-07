import argparse

import numpy as np
import json
import data
from collections import defaultdict
from data import normalize_answer
from bart import MyBartModel


class topKPassasages():
    """
    This class serves as modular way of retrieving top k passages of a question for reader
    """
    def __init__(self, k, passages_path, rank_path, data_path):
        # load wiki passages and store in dictionary
        self.rank = self.load_ranks(rank_path) # a list of lists of passsages ids   [ [ 3,5,9 ], ...  ]
        self.answers = self.load_answer(data_path) # {id:str, question:text, answer:text}
        self.passages = self.load_passages(passages_path) # a list of dictionary {title:str, text:str}
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
        return [self.passages[k] for r in self.ranks[i]]


    def topKRank(self, k=10):
        self.ranks = [r[:k] for r in self.ranks]

    def load_passages(self, passages_path):
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
        for d, passage_indices in zip(self.answers, self.rank):
            assert len(passage_indices)==100
            answers = [normalize_answer(answer) for answer in d["answer"]]
            passages = [normalize_answer(self.passages[passage_index]["text"]) for passage_index in passage_indices]
            for k in [1, 5, 10, 100]:
                recall[k].append(any([answer in passage for answer in answers for passage in passages[:k]]))

        for k in [1, 5, 10, 100]:
            print ("Recall @ %d\t%.1f%%" % (k, 100*np.mean(recall[k])))
        return recall


def select_passages(args):
    """
    Select top-k passages for each question and return the passages results in text format.
    Also, print the recall to evaluate the retriever.
    :param args:
    :return:
    """
    k = args.top_k
    wiki_passage_path = args.passages_path
    ranking_path = args.ranking_path
    data_path = args.data_path
    #
    # # load wiki passages and store in dictionary
    # wiki_data = []
    # with open(self.data_path, "rb") as f:
    #     for line in f.readlines():
    #         wiki_data.append(line.decode().strip().split("\t"))
    # assert wiki_data[0]==["id", "text", "title"]
    # import pdb
    # pdb.set_trace()
    # wiki_data = { {"title": title, "text": text} for _, text, title in wiki_data[1:]}


    ps = topKPassasages(k, wiki_passage_path, ranking_path, data_path)

    # pipeline
    # 1. select top 10 passages(up to 1024 tokens) for each question
    #           (do I calculate recall by using the answer label to evaluate retrieval quality? Yes. And compare it with the given results)
    # 2. conditioning on the question and top 10 passages output a representation
    # 3. select answer span that is most likely
    return ps


def predict(model, model_path=None):
    pass
    # concatenate dataset


    # input: [CLS] question [SEP] title 1 [SEP] passage1 [SEP] title 2 …. Passage 10
    # predict answer based on the dataset
    # Previous model only predict based on question (relying on model capacity)





    # return the best span

    # Q: do I use model from run?




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bart", type=str)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--ranking_path", default="data/reranking_results/nq_test.json")
    parser.add_argument("--passages_path", default="data/psgs_w100.tsv")
    parser.add_argument("--data_path", default="data/nqopen-test.json")
    parser.add_argument("--eval_recall", default=False)
    args = parser.parse_args()

    passages = select_passages(args)

    # Bart model and predict