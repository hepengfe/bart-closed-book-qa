# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# TODO: free memory of training data after dumping and load again after process eval data
# TODO: load embedding once

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch
import multiprocessing as mp
from run import run


def main():
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--train_file", default="data/nqopen-train.json")
    parser.add_argument("--predict_file", default="data/nqopen-dev.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_tokenize", action='store_true')
    parser.add_argument("--predict_type", default="thresholding", type=str)

    # Model parameters
    parser.add_argument("--model", type=str, default="bart")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--fine_tune", action="store_true" )
    parser.add_argument("--do_lowercase", action='store_true', default=True)


    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=32)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos",
                        action='store_true', default=False)
    parser.add_argument("--prepend_question_token", default=False)

    # data augumentation
    parser.add_argument("--augment_k_times", type = str, default="1", help= "can be 'varied' or int value")

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--start_epoch", default=0, type=int,
                        help="When restart from checkpoint, epoch will be overwritten to the correct one.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_parallel", type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda",
                        help="Can be set to cpu or cuda or device number")
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--gradient_cp',  default=False, action="store_true")

    # parameters for SpanSeqGen
    parser.add_argument("--top_k_passages", default=10, type=int)
    # "data/reranking_results/ambigqa"
    parser.add_argument("--ranking_folder_path",
                        default=None)  # "data/reranking_results/ambigqa"
    parser.add_argument("--data_folder_path", default=None)  # data/ambigqa
    parser.add_argument(
        "--passages_path", default="data/wiki/psgs_w100_20200201.tsv")  # psgs_w100.tsv
    parser.add_argument("--top_k_answers", default=1, type=int)
    parser.add_argument("--max_answer_length", default=10, type=int)

    parser.add_argument("--eval_recall", default=False)
    parser.add_argument("--threshold", type=int, default=0.1)


    # passage clustering
    parser.add_argument("--passage_clustering",
                        default=False, action="store_true")
    parser.add_argument("--k_cluster", default = 10, type=int) 
    parser.add_argument("--rank_threshold", default=60, type=int)
    parser.add_argument("--is_contrastive", default=False, action="store_true")


    # reset parameters
    parser.add_argument("--retokenize", default=False, action="store_true")
    parser.add_argument("--reencode", default=False, action="store_true")


    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    



    # contrastive
    if args.is_contrastive:
        assert args.passage_clustering == True, "PC must be enable to train a contrastive dataset"
    

    # Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.single_gpu = False
    if args.device == "cuda":  # use all gpus by default
        args.n_gpu = torch.cuda.device_count()
    elif args.device == "cpu":
        args.device = "cpu"
    else:
        # indicate it wants to use specific gpu
        args.device = int(args.device)
        args.n_gpu = 1

    if not args.do_train and not args.do_predict and not args.do_tokenize:
        raise ValueError(
            "At least one of `do_train` or `do_predict` or `do_tokenize` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError(
                "If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.model.lower() == "t5" and args.prepend_question_token == False:
        logger.warning("t5 model needs prepending, it's adjusted now")
        args.prepend_question_token = True
        
    logger.info("Using {} gpus".format(args.n_gpu))
    if args.device == "cuda":
        assert args.n_gpu > 1, "if there is only one gpu, set args.device=0"
    if args.do_predict:
        assert args.checkpoint is not None, "must have a model to load to make prediction"
    run(args, logger)


if __name__ == '__main__':
    main()
