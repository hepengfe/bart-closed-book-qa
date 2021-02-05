import os
import numpy as np
import torch
import json

from transformers import BartTokenizer, BartConfig, T5Tokenizer, T5Config
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from bart import MyBart, MyBartForCondGen
from T5 import MyT5

def run(args, logger):
    if args.model.lower() == "bart":
        tokenizer = BartTokenizer.from_pretrained("bart-large")
    elif args.model.lower() == "t5":
        # tokenizer = T5Tokenizer.from_pretrained("t5-large")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        print("wrong model argument")
        exit()
    train_data = QAData(logger, args, args.train_file, True)
    dev_data = QAData(logger, args, args.predict_file, False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()
    # args.device = 1
    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            if args.model.lower() == "bart":
                model = MyBart.from_pretrained("bart-large",
                                               state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
                # model = MyBartForCondGen.from_pretrained("bart-large",
                #                                          state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))

            elif args.model.lower() == "t5":
                # model = MyT5.from_pretrained('t5-large')
                model = MyT5.from_pretrained('t5-base')
            else:
                print("wrong model argument")
                exit()

        else:
            if args.model.lower() == "bart":
                model = MyBart.from_pretrained("bart-large")
                # model = MyBartForCondGen.from_pretrained("bart-large")
            elif args.model.lower() == "t5":
                # model = MyT5.from_pretrained('t5-large')
                model = MyT5.from_pretrained('t5-base')
            else:
                print("wrong model argument")
                exit()
            if args.device == "cuda" and torch.cuda.device_count() > 1:
                if args.n_gpu == 1:
                    logger.warning("User specified one gpu but there are actually {}, it has been corrected".format(torch.cuda.device_count()))
                    args.n_gpu = torch.cuda.device_count()
                model = torch.nn.DataParallel(model)



        # if args.device:

        model.to(torch.device(args.device))
        # elif torch.cuda.is_available():
        #     model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                                     num_warmup_steps=args.warmup_steps,
                                                     num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')

        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        model = MyBart.from_pretrained("bart-large",
                                       state_dict=convert_to_single_gpu(torch.load(checkpoint)))
        logger.info("Loading checkpoint from {}".format(checkpoint))
        # if args.single_gpu:
        model.to(torch.device(args.device))
        # elif torch.cuda.is_available():
        #     model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_data, args.predict_type , save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    epoch_losses = dict()
    epoch_ems = dict()
    best_accuracy = -1
    stop_training=False

    # reload some training status if
    if args.checkpoint is not None:
        with open(os.path.join(args.output_dir, 'checkpoint_stat.json'), "r") as fp:
            checkpoint_stat = json.load(fp)

        args.start_epoch = checkpoint_stat["best_epoch"]
        best_accuracy = checkpoint_stat["best_em_accuracy"]
        global_step = checkpoint_stat["global_step"]



    checkpoint_stat = dict()
    logger.info("Starting training!")
    for epoch in range(int(args.start_epoch), int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1


            # if args.single_gpu:
            batch = [b.to(torch.device(args.device)) for b in batch]
            # elif torch.cuda.is_available():
            #     batch = [b.to(torch.device("cuda")) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)

            # import pdb
            # pdb.set_trace()
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())

            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                # import pdb
                # pdb.set_trace()

                curr_em = inference(model if args.n_gpu==1 else model.module, dev_data, args.device, True)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em*100,
                    epoch))
                epoch_ems[epoch] = str(curr_em*100)
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    checkpoint_stat["best_epoch"] = epoch
                    checkpoint_stat["best_em_accuracy"] = curr_em
                    checkpoint_stat["global_step"] = global_step
                    with open(os.path.join(args.output_dir, 'checkpoint_stat.json'), "w") as fp:
                        json.dump(checkpoint_stat, fp)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
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
        with open(os.path.join(args.output_dir, f"{args.model}_bs{args.train_batch_size}.json") , "w") as fp:
            json.dump(epoch_losses, fp)
            json.dump(epoch_ems, fp)
        if stop_training:
            break

def inference(model, dev_data, predict_type, device = "cuda", save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id

    # if predict_type == "thresholding":
    #     # generate answer
    #
    # elif predict_type == "SpanSeqGen":
    #     print("not implemented yet")
    #     exit()

    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device(device)) for b in batch]
            import pdb
            pdb.set_trace()
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)

        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)

    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))


"""
python cli.py \
        --model bart \
        --do_predict \
        --predict_type thresholding \
        --output_dir out/nq-t5-closed-qa \
        --train_file data/nqopen-train.json \
        --predict_file data/nqopen-dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --device 0
        
python cli.py \
        --model bart \
        --do_predict \
        --predict_type thresholding \
        --output_dir out/nq-bart-closed-qa \
        --predict_file data/nqopen-dev.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --device 0
"""




