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
"""
train_bs=64
test_bs=64
python cli.py \
        --model t5 \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --device cuda



test----------------------------------------
train_bs=1
test_bs=1
CUDA_VISIBLE_DEVICES=1 python cli.py \
        --model t5 \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --device 0 \
        --gradient_cp True

test2-------------------------------------
train_bs=2
test_bs=2
python cli.py \
        --model bart \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --device cuda \
        --gradient_cp True





bart continue training-------------------
train_bs=130
test_bs=130
python cli.py
        --do_train --output_dir out/nq-bart-closed-qa \
        --model bart\
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --checkpoint out/nq-bart-closed-qa/best-model.pt \
        --device cuda \
        --gradient_cp False

t5 training-----------------------------------
# train_bs=180
# test_bs=180
# python cli.py \
#         --model t5 \
#         --do_train --output_dir out/nq-t5-closed-qa \
#         --train_file data/nqopen-train.json \
#         --predict_file data/nqopen-dev.json \
#         --train_batch_size ${train_bs} \
#         --predict_batch_size ${test_bs} \
#         --append_another_bos \
#         --device cuda \
#         --gradient_cp False


train_bs=180
test_bs=180
CUDA_VIDIABLE_DEVICES=0  python cli.py \
        --model t5 \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --device cuda \
        --gradient_cp False \
        --eval_period 1




--------------------To reproduce T5 bug
train_bs=250
test_bs=250
python cli.py \
        --model t5 \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --device 0 \
        --gradient_cp False \
        --eval_period 1000 


--------------------To generate text
train_bs=150
test_bs=150
python cli.py \
        --model bart \
        --do_train --output_dir out/nq-bart-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --device 0 \
        --gradient_cp False \
        --eval_period 1000 


----------------------debug what slows down the process in topKPassages
train_bs=150
test_bs=150
python cli.py \
        --model bart \
        --do_train --output_dir out/nq-bart-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 1024 \
        --top_k 1 \
        --eval_period 1000 


------- tokenization without training
python cli.py \
        --model bart \
        --do_tokenize --output_dir out/nq-bart-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 1024 \
        --top_k 1 \
        --eval_period 1000 


----------- train bart
python cli.py \
        --model bart \
        --do_train --output_dir out/nq-bart-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 750 \
        --top_k 5 \
        --eval_period 1000 

----------- train T5
python cli.py \
        --model t5 \
        --do_train --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 750 \
        --top_k 5 \
        --eval_period 1000 

-------- bart with dataset aimbig 
python cli.py \
        --model bart \
        --do_train --output_dir out/ambig-bart-closed-qa \
        --train_file data/ambigqa/ambigqa_train.json \
        --predict_file data/ambigqa/ambigqa_dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 750 \
        --top_k 5 \
        --eval_period 1
---------- span extract
python cli.py \
        --model bert \
        --do_train --output_dir out/np-bert-closed-qa \
        --train_file  data/nqopen/nqopen-train.json  \
        --predict_file data/nqopen/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanExtraction \
        --device 0 \
        --gradient_cp False \
        --max_input_length 1024 \
        --top_k 5 \
        --eval_period 1 \
        --ranking_folder_path data/reranking_results/nqopen \
        --data_folder_path data/nqopen \
        --passages_path data/wiki/psgs_w100.tsv \
        --debug
-----------
python cli.py \
        --model bert \
        --do_train --output_dir out/np-bert-closed-qa \
        --train_file  data/nqopen/nqopen-train.json  \
        --predict_file data/nqopen/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanExtraction \
        --device 0 \
        --gradient_cp False \
        --max_input_length 512 \
        --top_k 5 \
        --eval_period 1 \
        --ranking_folder_path data/reranking_results/nqopen \
        --data_folder_path data/nqopen \
        --passages_path data/wiki/psgs_w100.tsv
---------
python cli.py \
        --model bert \
        --do_train --output_dir out/np-bert-closed-qa \
        --train_file  data/nqopen/nqopen-train.json  \
        --predict_file data/nqopen/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanExtraction \
        --device cpu \
        --gradient_cp False \
        --max_input_length 750 \
        --top_k 5 \
        --eval_period 1 \
        --ranking_folder_path data/reranking_results/nqopen \
        --data_folder_path data/nqopen \
        --passages_path data/wiki/psgs_w100.tsv
---------
python cli.py \
        --model bert \
        --do_train --output_dir out/np-bert-closed-qa \
        --train_file  data/nqopen/nqopen-train.json  \
        --predict_file data/nqopen/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanExtraction \
        --device 0 \
        --gradient_cp False \
        --max_input_length 256 \
        --top_k 5 \
        --eval_period 1 \
        --ranking_folder_path data/reranking_results/nqopen \
        --data_folder_path data/nqopen \
        --passages_path data/wiki/psgs_w100.tsv
------- eval_period = 0
python cli.py \
        --model bert \
        --do_train --output_dir out/np-bert-closed-qa \
        --train_file  data/nqopen/nqopen-train.json  \
        --predict_file data/nqopen/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanExtraction \
        --device 0 \
        --gradient_cp False \
        --max_input_length 256 \
        --top_k 5 \
        --eval_period 0 \
        --ranking_folder_path data/reranking_results/nqopen \
        --data_folder_path data/nqopen \
        --passages_path data/wiki/psgs_w100.tsv
-------- Continue training Bart from model trained on NQ
python cli.py \
        --model bart \
        --do_train --output_dir out/ambig-bart-closed-qa \
        --train_file data/ambigqa/ambigqa_train.json \
        --predict_file data/ambigqa/ambigqa_dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --predict_type  SpanSeqGen \
        --device 0 \
        --gradient_cp False \
        --max_input_length 750 \
        --top_k 5 \
        --eval_period 1
"""


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

from run import run


def main():
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
    parser.add_argument("--do_lowercase", action='store_true', default=True)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=32)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos",
                        action='store_true', default=False)
    parser.add_argument("--prepend_question_token", default=False)

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
    parser.add_argument('--device', type=str, default="cuda",
                        help="Can be set to cpu or cuda or device number")
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--gradient_cp', type=bool, default=False)

    # parameters for SpanSeqGen
    parser.add_argument("--top_k", default=10, type=int)
    # "data/reranking_results/ambigqa"
    parser.add_argument("--ranking_folder_path",
                        default=None)  # "data/reranking_results/ambigqa"
    parser.add_argument("--data_folder_path", default=None)  # data/ambigqa
    parser.add_argument(
        "--passages_path", default="data/wiki/psgs_w100_20200201.tsv")  # psgs_w100.tsv

    parser.add_argument("--eval_recall", default=False)

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

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

        # args.device = "cuda"

    elif args.device == "cpu":
        args.device = "cpu"
    else:
        # indicate it wants to use specific gpu
        args.device = int(args.device)
        args.n_gpu = 1

    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
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
        logger.warning("t5 model needs prepending ")
        
    logger.info("Using {} gpus".format(args.n_gpu))
    run(args, logger)


if __name__ == '__main__':
    main()
