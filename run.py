from span_utils import decode
import os
from span_predictor import BertSpanPredictor, ElectraSpanPredictor
import numpy as np
import torch
import json

from transformers import BartTokenizer, BartConfig, T5Tokenizer, T5Config, BertConfig, BertTokenizer , ElectraConfig,  ElectraTokenizer,  ElectraForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from bart import MyBart 
from T5 import MyT5
from tqdm import tqdm



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
        model.to(torch.device(args.device))
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
                # # import pdb;
                # # pdb.set_trace()
                # # output = model(input_ids=batch[0], attention_mask=batch[1],
                # #              decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                # #              is_training=False)
                # print("check decoder input ids and input ids, check outputs without training mode as well")
                
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
                logger.info(f"Start evaluating at global step {global_step}")
                curr_em = inference(args, get_model(model, args.device), dev_data,
                                    args.predict_type, device=args.device, is_ambig=get_model(model, args.device).is_ambig, save_predictions=True)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em*100,
                    epoch))
                epoch_ems[epoch] = str(curr_em*100)
                train_losses = []
                if best_accuracy < curr_em:
                    get_model(model, args.device).save_pretrained(args.output_dir)
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

    if predict_type.lower() == "spanseqgen":
        for i, batch in tqdm(enumerate(dev_data.dataloader)) if args.verbose else enumerate(dev_data.dataloader):
            batch = [b.to(device) for b in batch]
            
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=dev_data.args.num_beams,
                                     max_length=dev_data.args.max_output_length,
                                     early_stopping=True,
                                     )
            
            # outputs = model.generate(input_ids=batch[0] )
            # Q: span efxtraction: what it generates?
            # overwrite bert generate function

            # TODO: if it's logits then use decode function in span utils
            for input_, output in zip(batch[0], outputs):
                
                pred = dev_data.decode(output)
                print("check prediction: ", pred)
                predictions.append(pred)
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
