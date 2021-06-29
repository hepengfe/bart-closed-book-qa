
import pdb
from re import A
from IPython import embed
from tqdm import tqdm

import numpy as np
import json
import os
import pickle # pickle is faster than json if the user doesn't need readability
import string
import re
import torch
import scipy
from scipy.special import log_softmax
from dateutil.parser import ParserError, parse
from number_parser import parse_number
import itertools
from collections import defaultdict


def some_generic_dump_pickle(d, path):
    with open(path, "wb") as fp:
        pickle.dump(d, fp)

def some_generic_load_pickle(d, path):
    with open(path, "rb") as fp:
        d = pickle.load(fp)
    return d


def eval(predictions, data, eval_fn, normaliza_fn,
                  dataset_name, answer_type) -> list:
    """ A generic evaluation function to evaluation predictions and data labels based on the provided evaluation function

    Args:
        predictions (list): it could be a list of predictions or a list of (prediction, prediction_score). It's checked and marked by is_pred_with_score
        data ([type]): [description]
        eval_fn ([type]): [description]
        normaliza_fn ([type]): [description]
        dataset_name ([type]): [description]
        answer_type ([type]): [description]

    Returns:
        list: [description]
    """
    eval_scores = []

    is_pred_with_score = False
    if type(predictions[0]) == tuple:
        is_pred_with_score = True
    for (prediction, dp) in zip(predictions, data):
        if dataset_name == "ambig":
            if answer_type == "seq":

                # useful when it's not parallel
                if type(predictions) == defaultdict:
                    # it's actually dictionary key, predictions is a dictionary here
                    question_idx = prediction
                    prediction = predictions[question_idx]
                cur_answers = dp["answers"]
                if is_pred_with_score:
                    prediction, score = prediction

                # f1 without duplication
                prediction = prediction.replace(
                    "<s>", "").replace("</s>", "").split("<sep>")
                prediction = [normaliza_fn(pred)
                                for pred in prediction]
                prediction = [pred for pred in prediction if len(
                    pred) != 0]  # remove empty prediction

                max_f1 = np.max([eval_fn(list(set(cur_answer)), list(set(prediction)))
                                    for cur_answer in cur_answers])
                if is_pred_with_score:
                    print(f"f1: {max_f1}  prediction: {prediction} pred_score: {score} cur_answer: {cur_answers[:10]}")
                else:
                    print(f"f1: {max_f1}  prediction: {prediction} cur_answer: {cur_answers[:10]}")
                # NOTE: the only difference from span answer type
                eval_scores.append(max_f1)

            elif answer_type == "span":
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
                        print("error in qa_d type: ")
                        exit()
                eval_scores.append(eval_fn(cur_answer, prediction))
        elif dataset_name == "nq":
            if answer_type == "seq":
                for (prediction, dp) in zip(predictions, data):
                    # there are many concatenation of answers and they are all correct
                    # we append the one with the highest score

                    eval_scores.append(eval_fn(
                        prediction, dp["answer"]))
            else:
                for (prediction, dp) in zip(predictions, data):
                    for pred in prediction:
                        eval_scores.append(eval_fn(pred, dp["answer"]))
    return eval_scores




def is_answer_set_in_passsages(answer_md, p_str, answers, remove_answer = False):
    """check if a passage contain any answer in the answer set

    Args:
        answer_md ([type]): [description]
        p_str ([type]): [description]
        answers ([type]): [description]
        remove_answer (bool): remove answer from matadata so as to 

    Returns:
        [type]: [description]
    """
    for cur_md_for_qa_pair in answer_md:
        for start, end in cur_md_for_qa_pair:
            answer_for_qa_pair = answers[start:end]
            for cur_a_str in answer_for_qa_pair:
                if is_answer_in_passages(cur_a_str, p_str):
                    if remove_answer:
                        answer_md.remove(cur_md_for_qa_pair)
                        return True, answer_md
                    else:
                        return True
    if remove_answer:
        return False, answer_md
    else:
        return False




def is_answer_in_passages(answer_str, p_str):
    """check the existance of answer in passages by comparing string

    Args:
        idx ([type]): [description]
    """
    return answer_str.lower() in p_str.lower()


def get_p_str(cur_qp, tokenizer, max_qp_length = None):
    if max_qp_length:
       qp_ids = tokenizer.encode(
           cur_qp)[:max_qp_length]
    else:
       qp_ids = tokenizer.encode(
           cur_qp)
    p_ids = qp_ids[qp_ids.index(tokenizer.eos_token_id):]
    p_str = tokenizer.convert_ids_to_tokens(p_ids)
    p_str = tokenizer.convert_tokens_to_string(
        p_str)
    return p_str

def concatenate_answers(answer_sets, sep_token):
    joined_answers = [answer for answer in itertools.product(*
                                                                answer_sets)]
    concatenated_answers = [sep_token.join(
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
        answer_str = answer_str.replace(",", "")
        if parse_number(answer_str) != None:  # example: 55,000
            return True
        return False  # example: more than 1 million
    except TypeError:
        return False


def preprocess_qpa(questions, question_ids, passages, answers, metadata, data,
                    top_k_passages,  tokenizer, 
                   answer_type, is_training, is_ambig, args,
                   logging_prefix, logger,
                   rank_threshold=None, clustered_passages_path=None) -> dict():

    
    # TODO: test ambig bart first
    # TODO: dump dictionary
    # TODO: parallel evaluate 
    # TODO: T5 models evaluate
    # TODO: BART debug training see the performance
    # TODO: number of pos examples and number of neg examples
    # TODO: check where the 5gb GPU memory comes from by setting pdb
        # something in training mode (as in prediciton mode there is no such a memory)

    # TODO: contrastive
        # encode file name   (add contrastive, if not contrastive, don't add contrastive (keep it the same))
        # tokenize file name 
        # each cluster will have at most one positive example and at most one negative example 
        # 
    # 
    # TODO: provide more clustering analytics. Given a question, how is the clustering? 
        # top-k passages contain the answer and top-k passages doesn't contain the answer.



    # -ambig(or say multi-answer dataset)
    # --PC
    # ---contrastive
    # ---non-contrastive
    qpa_dict = dict() # questions, answers, question_metadata, answer_metadata, data
    clustering_results = dict()
    num_clusters = 0
    num_passages = 0
    questions_with_clustered_passages = []
    if args.passage_clustering:
        assert is_ambig == True, "PC mode: must be for ambig or multi-answer datasets"
        assert rank_threshold is not None, "PC mode: there should be a PC rank threhold."
        assert clustered_passages_path is not None, "PC mode: there should be a clustered_passages_path"

    sep_token = "<SEP>"
    spaced_sep_token = " " + sep_token + " "

    question_metadata = []
    joined_answers_l = []
    empty_answer_str = "<s> </s>"
    

    if not is_ambig: # nq dataset
        for i in tqdm(range(len(questions))):
            # mark the begining of passages
            # end of question and start of passages
            questions[i] += " </s>  <s>"
            # add passage one by one
            for p in passages.get_passages(i, top_k_passages):
                # format: [CLS] question [SEP] title 1 [SEP] passages
                questions[i] += spaced_sep_token + \
                    p["title"] + spaced_sep_token + p["text"]
            # mark the begining of passages
            questions[i] += " </s> "
    else:
        if args.passage_clustering:
            logger.info(
                logging_prefix + "Concatenating clustering results...")
            assert len(question_ids) == len(
                metadata), (len(question_ids), len(metadata))
            # check answer distribution across top PCs, expecting most answers are in top clusters
            # by "exclusive", after we found one answer, we eliminate it. In this way, we can check the answer coverage by PC, and know how many clusters are necessary
            exclusive_answer_distribution_d = defaultdict(lambda :0)

            # check answer distribution without eliminating 
            answer_distribution_d = defaultdict(lambda: 0)

            # top PC number of different titles. Expecting number of titles are less in the top PC.
            # on average the numer of title in each PC. expect less title in top clusters (high rank, more accurate)
            title_distribution_d = defaultdict(lambda: 0)
            for (i, cur_md) in enumerate(metadata):
                clusters_passages, num_cluster_for_question_i, num_passages_for_question_i = passages.get_clustered_passages(
                    i, rank_threshold)  # 2-d list
                num_clusters += num_cluster_for_question_i
                num_passages += num_passages_for_question_i
                # make questions[i] a list, put index 0 a concatenation of all passsages
                # we want all passages because we want a joined_answer list for evaluation
                # Problem: they are not constrained by max_input_length correctly 
                # and are not the actual input


                if args.is_contrastive:
                    questions_with_clustered_passages.append(dict())
                    qp_d = questions_with_clustered_passages[-1]
                    qp_d["pos"] = []
                    qp_d["neg"] = []
                    # 1. needs truncation here?  probably not, we can directly check. 
                    # 2. check presence of answer. 
                    for p_cluster in clusters_passages:  # it's ordered
                        # reset qp concatenation
                        cluster_qp_concatenation = questions[i]
                        pos_cluster_qp_concatenation = cluster_qp_concatenation +  " </s>  <s>"
                        neg_cluster_qp_concatenation = cluster_qp_concatenation + " </s>  <s>"
                        pos_start = True
                        neg_start = True
                        for p in p_cluster:
                            # format: [CLS] question [SEP] title 1 [SEP] passages
                            if is_answer_set_in_passsages(cur_md, p["text"], answers):
                                if pos_start:
                                    pos_cluster_qp_concatenation += p["title"] + \
                                        spaced_sep_token + \
                                        p["text"]
                                    pos_start = False
                                else:
                                    pos_cluster_qp_concatenation += spaced_sep_token + \
                                        p["title"] + \
                                        spaced_sep_token + \
                                        p["text"]
                            else:
                                if neg_start:
                                    neg_cluster_qp_concatenation += p["title"] + \
                                        spaced_sep_token + \
                                        p["text"]
                                    neg_start = False
                                else:
                                    neg_cluster_qp_concatenation += spaced_sep_token + \
                                        p["title"] + \
                                        spaced_sep_token + \
                                        p["text"]
                        pos_cluster_qp_concatenation += " </s> "
                        neg_cluster_qp_concatenation += " </s> "
                        qp_d["pos"].append(
                            pos_cluster_qp_concatenation)
                        qp_d["neg"].append(
                            neg_cluster_qp_concatenation)
                else:
                    questions_with_clustered_passages.append([])
                    qp_l = questions_with_clustered_passages[-1]
                     
                    import copy
                    
                    updated_md = copy.deepcopy(cur_md)
                    for (i, p_cluster) in enumerate(clusters_passages):  # it's ordered
                        # reset qp concatenation
                        cluster_qp_concatenation = questions[i]
                        cluster_qp_concatenation += " </s>  <s>"
                        title_distribution_d[i] += len(set([p["title"] for p in p_cluster]))
                        start = True
                        for p in p_cluster:

                            # updated md
                            found_answer, updated_md = is_answer_set_in_passsages(
                                updated_md, p["text"], answers,True)
                            if found_answer:
                                exclusive_answer_distribution_d[i] += 1
                            
                            found_answer = is_answer_set_in_passsages(
                                cur_md, p["text"], answers)
                            if found_answer:
                                answer_distribution_d[i] += 1

                            
                                
                            # format: [CLS] question [SEP] title 1 [SEP] passages
                            if start:
                                cluster_qp_concatenation += p["title"] + \
                                    spaced_sep_token + \
                                        p["text"]
                            else:
                                cluster_qp_concatenation += spaced_sep_token + \
                            p["title"] + \
                                spaced_sep_token + \
                                    p["text"]
                            start = False
                        cluster_qp_concatenation += " </s> "
                        qp_l.append(
                            cluster_qp_concatenation)

            for i in range(num_clusters):
                title_distribution_d[i] /= len(metadata)
            if args.check:
                import pdb; pdb.set_trace()
            print("check title_distribution_d and answer_distribution_d and questions_with_clustered_passages")
            num_questions = len(questions)
            clustering_results = dict()
            clustering_results["num_clusters"] = num_clusters
            clustering_results["num_passages"] = num_passages
            clustering_results["num_questions"] = num_questions
            clustering_results["questions_n_passages"] = questions_with_clustered_passages
            with open(clustered_passages_path, "wb") as fp:
                pickle.dump(clustering_results, fp)
            # mark the begining of passages
            questions_n_passages = questions_with_clustered_passages
            logger.info(
                f"Average number of clusters is (better be around 2): {num_clusters/num_questions}")
            logger.info(
                f"Avg num of passages per cluster: {num_passages/num_clusters}")
        else: # non-clustering
            for i in tqdm(range(len(questions))):
                questions_n_passages = questions
                questions_n_passages[i] += " </s>  <s>" # end of question and start of passages
                # add passage one by one
                start = True
                # NOTE: get passage clustering
                for p in passages.get_passages(i, args.top_k_passages):
                    # format: [CLS] question [SEP] title 1 [SEP] passages
                    
                    if start:
                        questions_n_passages[i] += p["title"] + \
                            spaced_sep_token + p["text"]
                    else:
                        questions_n_passages[i] +=  spaced_sep_token + \
                            p["title"] +  spaced_sep_token + p["text"]
                    start = False
            questions_n_passages[i] += " </s> "

            # process answer for qp pair  (answer must be present in qp pair)
            # Q, P, A must be processed at the same time because they are affecting each other in the training setting


        new_questions = []
        
        new_answers = []
        answer_metadata = []
        question_indices = []
        eos_token_id = tokenizer.eos_token_id

        num_eliminated_qp = 0
        answer_presence_d = defaultdict(lambda: 0)
        # format QP and A
        for idx, (cur_qp, cur_md) in enumerate(zip(questions_n_passages, metadata)):
            
            found_answers_for_one_question = []
            # check existance of answers for latter joining (for evaluation)
            for cur_md_for_qa_pair in cur_md:
                found_answer_for_qa_pair = []
                # iterate acceptable answer (semantically similar answers)
                for start, end in cur_md_for_qa_pair:
                    # acceptable answers for one qa pair
                    answer_for_qa_pair = answers[start:end]
                    for cur_a_str in answer_for_qa_pair:
                        if args.passage_clustering:
                            # cur_qp is a list
                            # iterate all qp str (the actual input qp str)
                            for cur_qp_str in cur_qp["pos"] if args.is_contrastive else cur_qp:
                                p_str = get_p_str(cur_qp_str, tokenizer,
                                                args.max_input_length)
                                
                                if is_training and not args.debug:
                                    if is_answer_in_passages(cur_a_str, p_str):
                                        found_answer_for_qa_pair.append(
                                            cur_a_str)
                                else:  # add all answers in eval dataset or any dataset in debug mode no matter its presence in passages
                                    found_answer_for_qa_pair.append(
                                        cur_a_str)
                        else:
                            # cur_qp is single
                            p_str = get_p_str(cur_qp, tokenizer,
                                            args.max_input_length)

                            if is_training and not args.debug:
                                if is_answer_in_passages(cur_a_str, p_str):
                                    found_answer_for_qa_pair.append(
                                        cur_a_str)
                            else:  # add all answers in eval dataset or any dataset in debug mode no matter its presence in passages
                                found_answer_for_qa_pair.append(
                                    cur_a_str)
                if len(found_answer_for_qa_pair) > 0:
                    found_answers_for_one_question.append(
                        list(set(found_answer_for_qa_pair)))  # NOTE: remove duplicated answers for one qa pair

            if len(found_answers_for_one_question) == 0 and is_training:
                # actually in dev mode, length is certainly larger than zero as we will add answer no matter its presence in passages
                continue
            
            
            # NOTE: for regular training mode(no passage clustering), we still add answers for every question
            # add answers separately for each clusters for training + passage clustering mode
            if is_training and args.passage_clustering:
                # new type of
                # is_training -> seprate pairs of QP and A
                # not is_training -> combine as we used to do
                for (cluster_rank, cur_qp_str) in enumerate(cur_qp["pos"] if args.is_contrastive else cur_qp):
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
                            found_answers_for_one_qp.append(
                                found_answer_for_qa_pair)
                    # skip adding aligned questions and answers if not found qp
                    # do not add emtpy answer if it's contrastive learning
                    if len(found_answers_for_one_qp) == 0 and is_training and not args.is_contrastive:
                        num_eliminated_qp += 1  # not actually eliminated because
                        empty_answer_gen_ratio = 0.3
                        if np.random.rand() < empty_answer_gen_ratio: # randomly eliminate qp for empty answer
                            found_answer_for_qa_pair.append(
                                empty_answer_str)
                            found_answers_for_one_qp.append(
                                found_answer_for_qa_pair) # NOTE: why it's added even it's empty
                        else:
                            continue

                    aug_times += len(found_answers_for_one_qp)
                    aug_times += num_date
                    aug_times += num_long_answer
                    # concatenate qp's answers
                    cur_answers = concatenate_answers(
                        found_answers_for_one_qp, sep_token)
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
                            (question_start_idx, question_end_idx))  # we actually added just a qp pair
                        answer_metadata.append(
                            (answer_start_idx, answer_end_idx))
                        
                # NOTE: we only add one data enty with PC and the answer presented in the PC
                if args.is_contrastive and is_training:
                    for (cluster_rank, cur_qp_str) in enumerate(cur_qp["neg"]): 
                        
                        answer_start_idx = len(new_answers)
                        # maintain its 1-D format
                        new_answers.append(empty_answer_str)
                        answer_end_idx = len(new_answers)

                        question_start_idx = len(new_questions)
                        new_questions.append(cur_qp_str)
                        question_end_idx = len(new_questions)

                        question_metadata.append(
                            (question_start_idx, question_end_idx))  # we actually added just a qp pair
                        answer_metadata.append(
                            (answer_start_idx, answer_end_idx))



            else:  # add concatenation of answers in eval dataset
                joined_answers = [answer for answer in itertools.product(*found_answers_for_one_question)]
                joined_answers_l.append(joined_answers)

                concatenated_answers = [sep_token.join(
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


                # add question indices and metadata
                if args.is_contrastive:
                    # pos answer
                    answer_start_idx = len(new_answers)
                    # maintain its 1-D format
                    new_answers.extend(cur_answers)
                    answer_end_idx = len(new_answers)

                    question_start_idx = len(new_questions)
                    # rename for some clarity
                    question_id = question_ids[idx]

                    # check cluster passages
                    new_questions.extend(cur_qp["pos"])
                    question_indices.extend(
                        [question_id] * len(cur_qp["pos"]))

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

    
                    # neg answers
                    answer_start_idx = len(new_answers)
                    # maintain its 1-D format
                    new_answers.append(empty_answer_str) 
                    # it won't be part of evaluation actually (will be normalized or say elimiated)
                    # but it helps appending questions wtih false positve passage that's appended 
                    answer_end_idx = len(new_answers)

                    question_start_idx = len(new_questions)
                    # rename for some clarity
                    question_id = question_ids[idx]

                    # check cluster passages
                    new_questions.extend(cur_qp["neg"])
                    question_indices.extend(
                        [question_id] * len(cur_qp["neg"]))

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
                else:
                    answer_start_idx = len(new_answers)
                    # maintain its 1-D format
                    new_answers.extend(cur_answers)
                    answer_end_idx = len(new_answers)

                    question_start_idx = len(new_questions)
                    # rename for some clarity
                    question_id = question_ids[idx]
                    if args.passage_clustering:
                        # check cluster passages
                        new_questions.extend(cur_qp)
                        question_indices.extend(
                            [question_id] * len(cur_qp))
                    else:
                        new_questions.append(cur_qp)  #
                        question_indices.append(question_id)
                    # import pdb; pdb.set_trace()
                    # print("check first new_questions ")
                    question_end_idx = len(new_questions)
                    assert len(new_questions) == len(
                        question_indices), "length should be the same"
                    # TODO: find a way to save the question ids

                    question_metadata.append(
                        (question_start_idx, question_end_idx))
                    answer_metadata.append(
                        (answer_start_idx, answer_end_idx))
        
        import pdb; pdb.set_trace()

        if args.passage_clustering:
            import pdb
            pdb.set_trace()
            print("check answer_presence_d")
            print("check num_eliminated_qp ")

            logger.info(
                logging_prefix + f"Selected qp ratio: {len(question_metadata)/len(questions_n_passages)}")
            logger.info(
                logging_prefix + f"num_eliminated_qp")

        question_ids = question_indices
            # print("check question_ids set length")
            # import pdb; pdb.set_trace()
        questions = new_questions
        answers = new_answers
        # import pdb; pdb.set_trace()
        print("answers example: ", answers[:30])
        for (idx, joined_answers) in enumerate(joined_answers_l):
            data[idx]["answers"] = joined_answers

    qpa_dict["qp"] = questions
    qpa_dict["question_ids"] = question_ids
    qpa_dict["answers"] = answers
    qpa_dict["question_metadata"] = question_metadata
    qpa_dict["answer_metadata"] = answer_metadata
    qpa_dict["joined_answers_l"] = joined_answers_l
    qpa_dict["data"] = data
    return qpa_dict


def dump_pickle(input_data, question_metadata, question_ids, answer_data, answer_metadata, joined_answers, encoded_input_path):
         
    d = dict()
    d["encoded_input"] = input_data
    d["question_metadata"] = question_metadata
    d["answer_data"] = answer_data 
    d["answer_metadata"] = answer_metadata
    d["question_ids"] = question_ids
    d["joined_answers"] = joined_answers 
    processed_data_path = encoded_input_path.replace("_input", "_data")

    with open(processed_data_path, "wb") as fp:
        pickle.dump(d, fp)
   

    # question_metadata_path = metadata_path.replace(
    #     "metadata", "question_metadata")
    # answer_metadata_path = metadata_path.replace("metadata", "answer_metadata")




    # with open(encoded_input_path, "wb") as fp:
    #     pickle.dump(input_data, fp)
    # with open(question_metadata_path, "wb") as fp:
    #     pickle.dump(question_metadata, fp)
    # with open(encoded_answer_path, "wb") as fp:
    #     pickle.dump(answer_data, fp) 
    # with open(answer_metadata_path, "wb") as fp:
    #     pickle.dump(answer_metadata, fp)
    # with open(processed_data_path, "wb") as fp:
    #     pickle.dump(processed_data, fp)


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
    
def load_pickle(encoded_input_path): # , encoded_answer_path, metadata_path):
    """ load encoded input data (concatenations of question and passages) and answer data from picle files 

    Args:
        encoded_input_path ([type]): [description]
        encoded_answer_path ([type]): [description]

    Returns:
        [type]: [description]
    """


    # question_metadata_path = metadata_path.replace(
    #     "metadata", "question_metadata")
    # answer_metadata_path = metadata_path.replace("metadata", "answer_metadata")
    processed_data_path = encoded_input_path.replace("_input", "_data")
    with open(processed_data_path, "rb") as fp:
        d = pickle.load(fp)
    input_data = d["encoded_input"] 
    question_metadata = d["question_metadata"] 
    answer_data = d["answer_data"] 
    answer_metadata = d["answer_metadata"]
    question_ids = d["question_ids"]
    joined_answers = d["joined_answers"]
    # , processed_data
    return input_data, question_metadata, question_ids, answer_data, answer_metadata, joined_answers


    # with open(encoded_input_path, "rb") as fp:
    #     input_data = pickle.load(fp)
    # with open(encoded_answer_path, "rb") as fp:
    #     answer_data = pickle.load(fp)
    # if os.path.exists(question_metadata_path):
    #     with open(question_metadata_path, "rb") as fp:
    #         question_metadata = pickle.load(fp)
    # else:
    #     question_metadata = None 
    # with open(answer_metadata_path, "rb") as fp:
    #     answer_metadata = pickle.load(fp)
    # if os.path.exists(processed_data_path):
    #     with open(processed_data_path, "rb") as fp:
    #         processed_data = pickle.load(fp)
    # else:
    #     processed_data = None

    # return input_data, question_metadata, answer_data, answer_metadata, processed_data


def preprocess_span_input(encoded_input_path, encoded_answer_path, metadata_path, logger,  tokenizer, max_input_length, max_n_answers=1, questions=None, answers=None, metadata=None, all_titles=None, all_passages=None,
               is_training = True):
    
    """


    Args:
        encoded_input_path ([type]): [description]
        encoded_answer_path ([type]): [description]
        tokenizer ([type]): [description]
        max_input_length ([type]): [description]
        max_n_answers ([type]): the top n answers kept 
        questions ([type], optional): [description]. Defaults to None.
        answers ([type], optional): [description]. Defaults to None.
        metadata ([type], optional): [description]. Defaults to None.
        all_titles ([type], optional): [description]. Defaults to None.
        all_passages ([type], optional): [description]. Defaults to None.
        is_training (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    '''

    Genereate a list of passage containing the answer.

    tokenizer: bertTokenizer
    questions: a list of strings
    answers: a list of lists of strings (note: OK for NQ, but need modification for AmbigQA)
    metadata: start of the answer in the flattened list: curr_metadata[0] / end of the answer in the flattened list:  curr_metadata[1]
    all_titles: a list of lists of k strings, each question has a list of string titles
    all_passages: a list of lists of k strings
    '''

    
    if os.path.exists(encoded_input_path) and os.path.exists(encoded_answer_path) and os.path.exists(metadata_path):
        input_data, answer_data, metadata = load_pickle(
            encoded_input_path, encoded_answer_path, metadata_path)

        
    else:  # it also handles special case that there is no 
        assert questions != None, "There doesn't exist encoded path, users should pass input data as argument into preprocess_span_input"
        assert len(questions) == len(all_titles) == len(
            all_passages) == len(metadata)
        inputs = []
        # import pdb
        # pdb.set_trace()

        for question, titles, passages in tqdm(zip(questions, all_titles, all_passages)):

            concatenated_context = ""
            for title, passage in zip(titles, passages):
                if len(concatenated_context) > 0:
                    concatenated_context += " [SEP] "
                concatenated_context += title + " [SEP] " + passage
            inputs.append((question, concatenated_context))

        contained = []
        import pdb
        pdb.set_trace()
        for input, (s, e) in zip(inputs, metadata):
            
            curr_answers = answers[s:e]

            contained.append(any([answer.lower() in input[1].lower()
                                for answer in curr_answers])) # for all acceptable answers, if it's in input[1], it's represented as 1 in list
        logger.info(f"Top k passages contians {np.mean(contained)} answers")
        logger.info("Not found encoded cache, now encoding QP concatenation ")
        # encoding is time consuming part, this version of transformer doesn't have BartTokenizerFast
        input_data = tokenizer.batch_encode_plus(inputs, padding="max_length", max_length=max_input_length,
                                                truncation=True, return_attention_mask = True, return_token_type_ids = True, verbose = True)
        answer_data = tokenizer.batch_encode_plus(answers, verbose=True)
        
        dump_pickle(input_data, answer_data, metadata, encoded_input_path,
                    encoded_answer_path, metadata_path)

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    token_type_ids = input_data["token_type_ids"]

    # as some of input ids will be skipped 
    new_input_ids = []
    new_attention_mask = []
    new_token_type_ids = []

    start_positions = []
    end_positions = []
    answer_mask = []

    for idx, (curr_input_ids, curr_attention_mask, curr_token_type_ids, curr_metadata) in enumerate(zip(
                input_ids, attention_mask, token_type_ids, metadata)):





        # offset record sep token location in original text
        # Q: why it returns only one index? 
        # A: [...].index(<SEP> id)

        # first <SEP> is where passage starts
        # <s> 101
        # <\s> 102
        offset = 1 + curr_input_ids.index(tokenizer.sep_token_id)
        


        # NOTE: [1:-1] is to slice out <SEP> token
        # ids from matadata like (0,1) will omitted due to slicing
        answer_input_ids = [answer_data["input_ids"][i][1:-1] for i in range(curr_metadata[0], curr_metadata[1])]

        # now, detect answer spans from passages
        # span is represented by the indices in QP concatenation
        detected_spans = []
        # compare answer ids and passage ids(whose index starts with offset)
        for curr_answer_input_ids in answer_input_ids: # iterate acceptable answers
            for i in range(offset, len(curr_input_ids)-len(curr_answer_input_ids)+1): # scan through passage token ids
                if curr_input_ids[i:i+len(curr_answer_input_ids)]==curr_answer_input_ids: # window size is the length of the answer
                    detected_spans.append( (i, i+len(curr_answer_input_ids)-1))
                    if len(detected_spans)==max_n_answers:
                        break
        # during training, we skip data entry with no detected span
        # during inference stage, we still want to keep them for evaluation and all data should be included
        if is_training and len(detected_spans) == 0:
            continue



        # TODO: it could have some better way to save RAM but current implementation ensures correctness

        # NOTE: add into output data if there is detected spans
        new_input_ids.append(curr_input_ids)
        new_attention_mask.append(curr_attention_mask)
        new_token_type_ids.append(curr_token_type_ids) 
        start_positions.append([s[0] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        end_positions.append([s[1] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        answer_mask.append([1 for _ in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
    

    answer_coverage_rate = len(new_input_ids)/len(input_ids) # measure how often answers appear in passages
    return {"input_ids": new_input_ids, "attention_mask": new_attention_mask, "token_type_ids": new_token_type_ids,
            "start_positions": start_positions, "end_positions": end_positions, "answer_mask": answer_mask, "answer_coverage_rate": answer_coverage_rate}
import torch.nn.functional as F
def decode(start_logits, end_logits, input_ids, tokenizer, top_k_answers, max_answer_length, threshold,  is_ambig=False):
    """[summary]
    Decode start and end logits (span prediction) into a list of text and its score
    Args:
        start_logits (tensor): [description]
        end_logits ([type]): [description]
        input_ids ([type]): [description]
        tokenizer ([type]): [description]
        top_k_answers ([type]): [description]
        max_answer_length ([type]): [description]

    Returns:
        [type]: [description]
    """             
    assert len(start_logits)==len(end_logits)==len(input_ids)
    assert top_k_answers >= 3
    
    def to_log_softmax_scores(scores):
        ls_scores = [t[1] for t in scores]
        ls_scores = log_softmax(ls_scores).tolist()
        new_scores = []
        for i in range(len(scores)):
            new_t = (scores[i][0], ls_scores[i])
            new_scores.append(new_t)
        return new_scores

    all_predictions = []
    # loop over all questions (curr_input)
    for curr_start_logits, curr_end_logits, curr_input_ids in \
            zip(start_logits, end_logits, input_ids):

        assert len(curr_start_logits)==len(curr_end_logits)==len(curr_input_ids)
        # import pdb
        # pdb.set_trace()
        # print("check curr_input_ids and it should be a list")
        offset = 1 + curr_input_ids.index(tokenizer.sep_token_id)  
          
        curr_start_logits = curr_start_logits[offset:]
        curr_end_logits = curr_end_logits[offset:]
        scores = []
        for (i, s) in enumerate(curr_start_logits):
            for (j, e) in enumerate(curr_end_logits[i:i+max_answer_length]):
                scores.append(((i, i+j), s+e))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = to_log_softmax_scores(scores)
        # print("sorting finished")
        chosen_span_intervals = []
        nbest = [] # record n best for current question
        p_l = []  

        # postprocess spans and scores
        for (start_index, end_index), score in scores:
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            # skip on any overlapping indices
            # (start_index, (prev_start_index, prev_end_index), end_index)
            # (prev_start_index, (start_index, end_index), prev_end_index) 
            if any([start_index<=prev_start_index<=prev_end_index<=end_index or
                    prev_start_index<=start_index<=end_index<=prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals]):
                continue
            
            # input ids = batch sz x seq length
            # NOTE: I think the bug is because inputs have two answers. Just something causes it has different shape.
            # print("check input ids and it should be a list of int")
            answer_text = tokenizer.decode(
                curr_input_ids[offset+start_index:offset+end_index+1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).strip()
            nbest.append({
                'text': answer_text,
                'log_softmax': score})

            if len(nbest)==top_k_answers:
                break
        
        if is_ambig:
            included = set()
            _nbest = []
            for p in nbest:
                text = normalize_answer(p["text"])
                if text not in included:
                    _nbest.append(p)
                    included.add(text)
            nbest = _nbest
            threshold = 0.1
            _nbest = [p for p in nbest if p['log_softmax'] >= np.log(threshold)] # TODO: threshold
            for p in nbest:
                p_l.append(p['log_softmax'])
            
            if len(_nbest) == 0:
                nbest = [nbest[0]]
            elif len(_nbest) > 5:
                nbest = _nbest[:5]
            else:
                nbest = _nbest
        print(np.mean(p_l))
        # a list of dictionary
        all_predictions.append(nbest)

    text_predictions = []
    for preds in all_predictions:
        text_predictions.append([ pred['text'] for pred in preds ])
    return text_predictions
