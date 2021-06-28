from span_utils import decode
import os
from span_predictor import BertSpanPredictor, ElectraSpanPredictor
import numpy as np
import torch
import json

from transformers import BartTokenizer, BartConfig, T5Tokenizer, T5Config, BertConfig, BertTokenizer , ElectraConfig,  ElectraTokenizer,  ElectraForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

from data3 import QAData
from bart import MyBart 
from T5 import MyT5
from tqdm import tqdm


def parallel_generate(model, device, input_ids, attention_mask, num_beams, max_output_length):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=num_beams,
                            max_length=max_output_length,
                            early_stopping=True,
                            use_cache=True
                            )
    return (device, outputs.to("cpu"))  # transfer output to device cpu, also might save some GPU memory


def parallel_decode(output, dev_data, index, q_id=None):

    pred = dev_data.decode(output)
    return (index, (q_id, pred))

def parallel_eval(eval_fn, partial_preds, partial_data):
    pass



def run(args, logger):

    
    # load tokenizer
    if args.predict_type.lower() == "spanseqgen":
        if args.model.lower() == "bart":
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        elif args.model.lower() == "t5":
            # tokenizer = T5Tokenizer.from_pretrained("t5-large")
            tokenizer = T5Tokenizer.from_pretrained("t5-base")

        else:
            print("wrong model argument")
            exit()
    elif args.predict_type.lower() == "spanextraction":
        if args.model.lower() == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif args.model.lower() == "electra":
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
        else:
            logger.warn(
                "Please specify correct model for span extraction. e.g. bert")
    else:
        print("wrong argument: ",  args.predict_type.lower())
        exit()



    if args.model.lower() == "bart" or args.model.lower() == "t5":
        logger.info("Add <sep> token into tokenizer")
        # add extra token for BART
        tokenizer.add_tokens(["<SEP>"], special_tokens=True)
        if tokenizer.bos_token_id == None:
            tokenizer.add_tokens(["<s>"], special_tokens=True)





    if args.do_tokenize:
        # during the process train_data will be overwritten, so memory will be collected
        for k in range(5, 15):
            for l in range(600, 800, 50):
                print("Evaluate passage coverage for top ", k,
                      "passages for max input sequence length ", l)
                args.top_k_passages = k
                args.max_input_length = l
                train_data = QAData(logger, args, args.train_file, "train")
                if args.do_predict:
                    dev_data = QAData(logger, args, args.predict_file, "test") 
                else:
                    dev_data = QAData(logger, args, args.predict_file, "dev")

                print("Pre-process training data")
                train_data.load_dataset(tokenizer)
                train_data.load_dataloader()

                print("Pre-process development data")
                dev_data.load_dataset(tokenizer)
                dev_data.load_dataloader()
        print("finished tokenization")
        exit()
    else:
        answer_type = "span" if "extraction" in args.predict_type.lower() else "seq"
        logger.info(f"answer type is {answer_type}")
        
        if args.do_predict:
            dev_data = QAData(logger, args, args.predict_file, "test") 
            dev_data_prefix = "[TEST DATA]\t" 
        else:
            # temp for memory trick
            # dev_data = QAData(logger, args, args.predict_file, "dev")
            # dev_data.load_dataset(tokenizer)
            # dev_data.load_dataloader()

            train_data = QAData(logger, args, args.train_file, "train") 
            dev_data = QAData(logger, args, args.predict_file, "dev")
            train_data_prefix =  "[TRAIN DATA]\t"
            logger.info(train_data_prefix + "Start loading...")
            logger.info(train_data_prefix + f"batch size {args.train_batch_size}")
            
            train_data.load_dataset(tokenizer)
            train_data.load_dataloader()
            dev_data_prefix = "[DEV DATA]\t" 
        logger.info(dev_data_prefix +"Start loading...")
        logger.info(dev_data_prefix + f"batch size {args.predict_batch_size}")

        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    model_prefix = f"[{args.model.upper()}]\t"
    if args.checkpoint is not None:

        if args.checkpoint.endswith(".pt"): # load old type of checkpoint
            logger.info(f"{model_prefix}Load old model with pt data format")
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key): value for key, value in state_dict.items()}
            if args.model.lower() == "bart":
                # TODO: add flag that when there is more specialized token,
                # NOTE: it serves a template to 

                # config = BartConfig.from_pretrained("bart-large")
                # config.gradient_checkpointing = args.gradient_cp
                config = BartConfig.from_pretrained("facebook/bart-large")
                logger.warn("Due to the previously added token, here I manually add one on config vocab size")
                # NOTE: old checkpoint doesn't save config, so it needs reload from scratch and modify 
                config.vocab_size += 1
                config.gradient_checkpointing = args.gradient_cp
                model = MyBart.from_pretrained(None,
                                                state_dict=convert_to_single_gpu(torch.load(args.checkpoint)), config = config)

            elif args.model.lower() == "t5":

                config = BartConfig.from_pretrained("t5-base")
                config.vocab_size += 2
                config.gradient_checkpointing = args.gradient_cp
                model = MyT5.from_pretrained(None, 
                                                state_dict=convert_to_single_gpu(torch.load(args.checkpoint)), config = config)
                logger.warn(
                    "Due to the previously added token, here I manually add one on config vocab size")
                
            elif args.model.lower() == "bert":
                
                model = BertSpanPredictor.from_pretrained(
                    "bert-base-uncased", state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
            elif args.model.lower() == "electra":
                # config = ElectraConfig.from_pretrained(args.checkpoint)
                # config.gradient_checkpointing = args.gradient_cp
                # model =  ElectraSpanPredictor.from_pretrained(args.checkpoint) 
                model = ElectraSpanPredictor.from_pretrained(
                    "electra-large-uncased", state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
            else:

                print("wrong model argument: ", args.model.lower())
                exit() 
        else: # load the new type of checkpoint
            logger.info(f"{model_prefix}Load checkpoint model")
            if args.model.lower() == "bart":
                # TODO: add flag that when there is more specialized token,
                # NOTE: it serves a template to 

                config = BartConfig.from_pretrained(args.checkpoint)
                config.gradient_checkpointing = args.gradient_cp


                # the other way is to save a bart-large file with resize token size 
                model = MyBart.from_pretrained(args.checkpoint, config = config)
            elif args.model.lower() == "t5":
                config = T5Config.from_pretrained(args.checkpoint)
                if args.gradient_cp:
                    logger.warn("T5 gradient checkpoint hasn't been implemented")
                    args.gradient_cp = False
                config.gradient_checkpointing = args.gradient_cp
                model =  MyT5.from_pretrained(args.checkpoint, config = config)

            elif args.model.lower() == "bert":
                config = BertConfig.from_pretrained(args.checkpoint)
                config.gradient_checkpointing = args.gradient_cp
                model =  BertSpanPredictor.from_pretrained(args.checkpoint, config = config)
            elif args.model.lower() == "electra":
                config = ElectraConfig.from_pretrained(args.checkpoint)
                config.gradient_checkpointing = args.gradient_cp
                model =  ElectraSpanPredictor.from_pretrained(args.checkpoint, config = config) 
            else:
                print("wrong model argument: ", args.model.lower())
                exit()
    is_ambig = False
    if args.fine_tune:
        is_ambig = True
        if args.model.lower() == "bert" or args.model.lower() == "electra":
            model.set_ambig(args.threshold)  # as it only affects generating, we set the class variable here
        else:
            model.set_ambig()

    if args.do_train:
        if args.checkpoint is None:
            logger.info(f"{model_prefix}gradient checkpoint mode:  {args.gradient_cp}")
            logger.info(f"{model_prefix}Loading pre-trained model ")
            # spanseqgen
            if args.predict_type.lower() == "spanseqgen":
                if args.model.lower() == "bart":
                    config = BartConfig.from_pretrained("facebook/bart-large")
                    config.gradient_checkpointing = args.gradient_cp
                    config.vocab_size += 1
                    model = MyBart.from_pretrained("facebook/bart-large", config=config)

                    # The new vector is added at the end of the embedding matrix
                    # set it to Randomly generated matrix
                    # as there is new token <SEP>
                    model.resize_token_embeddings(len(tokenizer))

                    # model.shared.weight[-1, :] = torch.zeros([model.config.hidden_size])

                elif args.model.lower() == "t5":
                    # model = MyT5.from_pretrained('t5-large')
                    config = T5Config.from_pretrained('t5-base')
                    config.vocab_size += 2
                    config.gradient_checkpointing = args.gradient_cp
                    model = MyT5.from_pretrained('t5-base')
                    model.resize_token_embeddings(len(tokenizer))
                else:
                    print("wrong model argument")
                    exit()
            # span extraction
            elif args.predict_type.lower() == "spanextraction":
                logger.info(f"{model_prefix}model enabled for span predictions")
                
                # TODO: add more variants span extraction pre-trained model
                if args.model.lower() == "bert":
                    config = BertConfig.from_pretrained("bert-base-uncased")
                    
                    config.gradient_checkpointing = args.gradient_cp
                    model = BertSpanPredictor.from_pretrained(
                        "bert-base-uncased", config=config)
                elif args.model.lower() == "electra":
                    config = ElectraConfig.from_pretrained("google/electra-large-discriminator")
                    config.gradient_checkpointing = args.gradient_cp
                    model = ElectraSpanPredictor.from_pretrained(
                        'google/electra-large-discriminator', config = config)
                    
                else:
                    logger.warn("Wrong model argument")
                    exit()
        # data parallel
        if args.device == "cuda" and torch.cuda.device_count() > 1:
            if args.n_gpu == 1:
                logger.warning("User specified one gpu but there are actually {}, it has been corrected".format(
                    torch.cuda.device_count()))
                args.n_gpu = torch.cuda.device_count()
            model = torch.nn.DataParallel(model)
            logger.info(f"{model_prefix}data parallelism status: True")
        # model parallism
        if args.device != "cuda":
            if args.model_parallel == True:
                logger.warn("only one gpu is enabled so model parallel is now disabled") 
                args.model_parallel = False
        if args.model_parallel and hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            model.is_model_parallel = True
        else:
            model.is_model_parallel = False        
        logger.info(f"{model_prefix}model parallelism status: {model.is_model_parallel}")
        model.to(torch.device(args.device))


        # training schedule and optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=100000)


        # start trianing
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)
    

    
    if args.do_predict:
        logger.info(f"[{args.model}] start prediction")
        
        
        model.eval()

        ems = inference(args, model, dev_data, args.predict_type,
                        device=args.device, is_ambig = is_ambig, save_predictions=True)
        logger.info("%s on %s data: %.2f" %
                    (dev_data.metric, dev_data.data_type, np.mean(ems)*100))


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    epoch_losses = dict()
    epoch_ems = dict()
    best_accuracy = -1
    stop_training = False

    # reload some training status if
    if args.fine_tune:
        assert args.checkpoint != None, "assert fine-tuning must have pre-trained model"

    if args.checkpoint is not None:
        if args.fine_tune:
            logger.info("Load previous model and start fine tuning on ambig dataset")
            logger.info(f"Augument {args.augment_k_times} times Ambig Questions")
            if args.augment_k_times != "varied":
                if args.augment_k_times.isdigit():
                    args.augment_k_times = int(args.augment_k_times)
                else:
                    raise NotImplementedError() 
        else:
            logger.info("Not continue fine tuning on Ambig, loading previous checkpoint stat and model")
            with open(os.path.join(args.output_dir, 'checkpoint_stat.json'), "r") as fp:
                checkpoint_stat = json.load(fp)

            start_epoch = checkpoint_stat["best_epoch"] 
            best_accuracy = checkpoint_stat["best_em_accuracy"]
            global_step = checkpoint_stat["global_step"] 
            logger.info(f"load checkpoint model successfully")
            logger.info(
                f"previous best model achieved {best_accuracy} at global_step {global_step} and epoch {start_epoch} ")
            # new start global step 
            global_step += args.eval_period

    checkpoint_stat = dict()
    logger.info(f"[{args.model}]   Start training!")
    epoch_range = range(int(args.start_epoch), int(args.num_train_epochs))
    epoch_range = tqdm(epoch_range) if args.verbose else epoch_range 
    wait_step = 0
    for epoch in epoch_range:   
        if args.verbose:
            logger.info(f"[{args.model}]\t epoch: {epoch}")
        for batch in tqdm(train_data.dataloader) if args.verbose else train_data.dataloader:
            global_step += 1

            batch = [b.to(args.device) for b in batch]
            if args.predict_type.lower() == "spanseqgen":
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                             is_training=True)
            elif args.predict_type.lower() == "spanextraction":
                loss = model(input_ids=batch[0],  attention_mask=batch[1],
                             token_type_ids=batch[2],
                             start_positions=batch[3], end_positions=batch[4], answer_mask=batch[5],
                             is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())

            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()


            # eval
            if global_step % args.eval_period == 0:
                model.eval()
                import pdb; pdb.set_trace()
                print("check what causes memory leaking")
                
                logger.info(f"Start evaluating at global step {global_step}")
                model = get_model(model, args.device).to("cpu")
                del batch
                # it will clean two things. 
                # 1. Model parameters(after model is moved to cpu)
                # 2. gradients info during training
                assert args.gradient_accumulation_steps == 1, "it's safe to empty cache only when gradient_accumulation_steps is one"
                torch.cuda.empty_cache()  
                # curr_em = inference(args, get_model(model, args.device), dev_data,
                #                     args.predict_type, device=args.device, is_ambig=get_model(model, args.device).is_ambig, save_predictions=True)
                curr_em = inference(args, model, dev_data,
                                    args.predict_type, device=args.device, is_ambig=model.is_ambig, save_predictions=True)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em*100,
                    epoch))
                epoch_ems[epoch] = str(curr_em*100)
                train_losses = []
                if best_accuracy < curr_em:
                    # get_model(model, args.device).save_pretrained(args.output_dir)
                    model.save_pretrained(
                        args.output_dir)
                    checkpoint_stat["best_epoch"] = epoch
                    checkpoint_stat["best_em_accuracy"] = curr_em
                    checkpoint_stat["global_step"] = global_step
                    with open(os.path.join(args.output_dir, 'checkpoint_stat.json'), "w") as fp:
                        json.dump(checkpoint_stat, fp)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model = torch.nn.DataParallel(model)
                model = model.to(torch.device(args.device))

                model.train()
            epoch_losses[epoch] = str(np.mean(train_losses))
        with open(os.path.join(args.output_dir, f"{args.model}_bs{args.train_batch_size}.json"), "w") as fp:
            json.dump(epoch_losses, fp)
            json.dump(epoch_ems, fp)
        if stop_training:
            break
def get_model(model, device):
    return model.module if device=="cuda" else model

def inference(args, model, dev_data, predict_type, device="cuda", is_ambig = False, save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    # import pdb; pdb.set_trace()
    # print("check dev data question ids")
    if predict_type.lower() == "spanseqgen":
        from collections import defaultdict
        # {question_idx : [pred_str1, pred_str2]}
        prediction_dict = defaultdict(lambda :[])


            # start_time = time.perf_counter()
            # outputs = model.generate(input_ids=input_ids,
            #                          attention_mask=attention_mask,
            #                          num_beams=dev_data.args.num_beams,
            #                          max_length=dev_data.args.max_output_length,
            #                          early_stopping=True,
            #                          use_cache = True
            #                          )
            # end_time = time.perf_counter()
            # print(f"Start Time : {start_time}")
            # print(f"End Time : {end_time}")
            # print(f"Execution Time : {end_time - start_time:0.6f}")

            # start_time = time.perf_counter()
            # outputs = model.generate(input_ids=input_ids,
            #                 attention_mask=attention_mask,
            #                 num_beams=dev_data.args.num_beams,
            #                 max_length=dev_data.args.max_output_length,
            #                 early_stopping=True,
            #                 )
            # end_time = time.perf_counter()
            # print(f"Start Time : {start_time}")
            # print(f"End Time : {end_time}")
            # print(f"Execution Time : {end_time - start_time:0.6f}"
        import copy
        if args.n_gpu > 1:
            model_on_devices = [copy.deepcopy(model).to(i) for i in range(args.n_gpu)]
            # model_on_devices = [copy.deepcopy(model).to(0), model.to(1)]
            # print([model.device for model in model_on_devices])
            # exit()
        import multiprocessing as mp
        for i, batch in tqdm(enumerate(dev_data.dataloader)) if args.verbose else enumerate(dev_data.dataloader):
            
            
            bs = len(batch[0])
            # assert bs % args.n_gpu == 0, "must be dividable?"


            # move model to device 0 and device 1
            # move input_ids, attention mask to device 0 and device 1
            # concatenate outputs
            # import pdb;pdb.set_trace()
         
            input_ids = batch[0]
            attention_mask = batch[1]
            question_ids = batch[2]

            
            if args.n_gpu > 1:
                
                # mp.set_start_method('spawn', force= True)
                # try:
                #     mp.set_start_method('spawn')
                # except RuntimeError:
                #     pass
                pool = mp.Pool(30)

                devices = list(range(args.n_gpu))
                bs_per_device = bs // args.n_gpu    
                # splitted_input_ids = [input_ids[i*bs_per_device: min((i+1)*bs_per_device, bs)]
                #                       for i in range(args.n_gpu)]

                # splitted_attention_mask = [
                #     attention_mask[i*bs_per_device: (i+1)*bs_per_device] for i in range(args.n_gpu)]
                splitted_input_ids = []
                splitted_attention_mask = []
                for i in range(args.n_gpu):
                    if i == args.n_gpu - 1:
                        splitted_input_ids.append(input_ids[i*bs_per_device:])
                        splitted_attention_mask.append(
                            attention_mask[i*bs_per_device: ])
                        break
                    splitted_input_ids.append(input_ids[i*bs_per_device: (i+1)*bs_per_device] )
                    splitted_attention_mask.append(
                        attention_mask[i*bs_per_device: (i+1)*bs_per_device])
                    
                        

                parallel_gen_input = zip(model_on_devices, devices, splitted_input_ids, splitted_attention_mask,
                                [dev_data.args.num_beams]*args.n_gpu, [dev_data.args.max_output_length]*args.n_gpu)
                indexed_outputs = dict(pool.starmap(
                    parallel_generate, parallel_gen_input))
                outputs = []
                for i in range(args.n_gpu):
                    outputs.extend(indexed_outputs[i].tolist()) # tensor to list 

                pool.close()
                pool.join()

            else:

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        num_beams=dev_data.args.num_beams,
                                        max_length=dev_data.args.max_output_length,
                                        early_stopping=True,
                                        use_cache = True,
                                        return_dict_in_generate = True,
                                        ).tolist()
            assert len(outputs) == len(question_ids) == len(
                attention_mask), (len(outputs), len(question_ids), len(attention_mask))
            if not args.passage_clustering:
                preds = dev_data.batch_decode(outputs)
                for pred in preds:
                    print("check prediction: ", pred)
                predictions.extend(preds)
            else:
                preds = dev_data.batch_decode(outputs)
                for (idx, q_id) in enumerate(question_ids):
                    try:
                        print(f"check prediction {q_id}: ", preds[idx])
                        prediction_dict[q_id].append(preds[idx])
                    except IndexError:
                        import pdb; pdb.set_trace()


        # second generation
        if args.passage_clustering and not args.is_contrastive:
            # remove empty string answers
            for q_id in prediction_dict.keys():
                prediction_dict[q_id] = [
                    a for a in prediction_dict[q_id] if len(a.strip()) != 0]


            # iterate all data again
            for i, batch in tqdm(enumerate(dev_data.dataloader)) if args.verbose else enumerate(dev_data.dataloader):
                # if i == 20:
                #     break
                input_ids = batch[0]
                attention_mask = batch[1]
                question_ids = batch[2]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                # new_input_ids = 
                indices = []
                qp_check_d = defaultdict(lambda :False)

                for i, (input_, q_id) in enumerate(zip(input_ids, question_ids)):
                    if len(prediction_dict[q_id]) == 0 and not qp_check_d[q_id]:
                        indices.append(i)
                        qp_check_d[q_id] = True
                indices = torch.LongTensor(indices)
                if len(indices) == 0:
                    continue
                
                # import pdb; pdb.set_trace()
                new_input_ids = input_ids[indices, :].to(device)
                new_attention_mask = attention_mask[indices, :].to(device)
                model = model.to(device)
                new_question_ids = []
                for idx in indices:
                    new_question_ids.append(question_ids[idx])
                new_question_ids = tuple(new_question_ids)
                 

                # disallow generate empty strings will prevent 
                # check if empty string id and sep token id is the same

                new_outputs = model.generate(input_ids=new_input_ids,
                                         attention_mask=new_attention_mask,
                                         num_beams=dev_data.args.num_beams,
                                         max_length=dev_data.args.max_output_length,
                                         early_stopping=True,
                                        bad_words_ids = [[2,0,0,1437, 2], [2,0,0,0]]
                                         )  # min_len =4  is about two words
                print(new_outputs)
                # filtered input ids
                for input_, output, q_id in zip(new_input_ids, new_outputs, new_question_ids):
                    pred = dev_data.decode(output)
                    print(f"check new prediction for question {q_id}: ", pred)
                    prediction_dict[q_id].append(pred)
                



        # PC eval: after all predictions
        if args.passage_clustering:
           
            print('check length of prediction dict keys')
            for i in prediction_dict.keys():
                preds = prediction_dict[i]  # predictions for one question
                preds = [p for p in preds if len(p) != 0]
                preds = "<sep>".join(preds)
                prediction_dict[i] = preds
            # import pdb
            # pdb.set_trace()
            # print("check predictions ")
        
            predictions = prediction_dict  # rename for convenince
            # import pdb
            # pdb.set_trace()
        print("check predict dict")

                
            # outputs = model.generate(input_ids=batch[0] )
            # Q: span efxtraction: what it generates?
            # overwrite bert generate function


        # another mothod is decode after all outputs
        # we don't enforce ordered prediction from each question
        # we decode and concatenate it based on question id  (dictionary)
        # finally 


    elif predict_type.lower() == "spanextraction":
        all_start_logits = []

        all_end_logits = []
        all_input_data = []

        for i, batch in tqdm(enumerate(dev_data.dataloader)) if args.verbose else enumerate(dev_data.dataloader):
            batch = [b.to(device) for b in batch]
            qa_outputs = model(input_ids=batch[0], attention_mask=batch[1],
                                             token_type_ids=batch[2], inputs_embeds=None,
                                             start_positions=batch[3], end_positions=batch[4], answer_mask=batch[5],
                                             is_training=False)
            start_logits, end_logits = qa_outputs.start_logits, qa_outputs.end_logits

            start_logits = start_logits.detach().cpu().numpy().tolist()
            end_logits = end_logits.detach().cpu().numpy().tolist()
            input_ids = batch[0].detach().cpu().numpy().tolist()
            all_input_data += input_ids
            all_start_logits += start_logits
            all_end_logits += end_logits

            

        predictions = decode(all_start_logits, all_end_logits, all_input_data, dev_data.tokenizer,
                         args.top_k_answers, max_answer_length=args.max_answer_length, threshold = args.threshold, is_ambig = is_ambig)

    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))
