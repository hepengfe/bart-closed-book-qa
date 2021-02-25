from span_utils import decode
import os
from span_predictor import SpanPredictor
import numpy as np
import torch
import json

from transformers import BartTokenizer, BartConfig, T5Tokenizer, T5Config, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from bart import MyBart, MyBartForCondGen
from T5 import MyT5

# TODO 1: ELECTRA
# TODO 2: 


def run(args, logger):

    if args.predict_type.lower() == "spanseqgen":
        if args.model.lower() == "bart":
            tokenizer = BartTokenizer.from_pretrained("bart-large")
            # tokenizer = BartTokenizer.from_pretrained("bart-large")
        elif args.model.lower() == "t5":
            # tokenizer = T5Tokenizer.from_pretrained("t5-large")
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
        else:
            print("wrong model argument")
            exit()
    elif args.predict_type.lower() == "spanextraction":
        if args.model.lower() == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        else:
            logger.warn(
                "Please specify correct model for span extraction. e.g. bert")
    else:
        print("wrong argument: ",  args.predict_type.lower())
        exit()

    if args.do_tokenize:
        # during the process train_data will be overwritten, so memory will be collected
        for k in range(5, 15):
            for l in range(600, 800, 50):
                print("Evaluate passage coverage for top ", k,
                      "passages for max input sequence length ", l)
                args.top_k = k
                args.max_input_length = l
                train_data = QAData(logger, args, args.train_file, "train")
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
        train_data = QAData(logger, args, args.train_file, "train")
        dev_data = QAData(logger, args, args.predict_file, "dev")

        print("Pre-process training data")
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()

        print("Pre-process development data")
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()
    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key): value for key, value in state_dict.items()}
            if args.model.lower() == "bart":
                # TODO: add flag that when there is more specialized token,

                model = MyBart.from_pretrained(
                    "bart-large",  state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))

                # model = MyBart.from_pretrained("bart-large",
                #                                state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
                # model = MyBartForCondGen.from_pretrained("bart-large",
                #                                          state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))

            elif args.model.lower() == "t5":
                # model = MyT5.from_pretrained('t5-large')
                model = MyT5.from_pretrained('t5-base')
            else:
                print("wrong model argument")
                exit()

        else:

            # spanseqgen
            if args.predict_type.lower() == "spanseqgen":
                if args.model.lower() == "bart":
                    # default_config = BartConfig.from_pretrained("bart-large")
                    # config = BartConfig.from_pretrained("bart-large", vocab_size = default_config.vocab_size + 1 )

                    config = BartConfig.from_pretrained("bart-large")
                    model = MyBart.from_pretrained("bart-large", config=config)

                    # The new vector is added at the end of the embedding matrix
                    # set it to Randomly generated matrix
                    # as there is new token <SEP>
                    model.resize_token_embeddings(len(tokenizer))

                    # model.model.shared.weight[-1, :] = torch.zeros([model.config.hidden_size])

                    # model = MyBartForCondGen.from_pretrained("bart-large")
                elif args.model.lower() == "t5":
                    # model = MyT5.from_pretrained('t5-large')
                    model = MyT5.from_pretrained('t5-base')
                else:
                    print("wrong model argument")
                    exit()
            # span extraction
            elif args.predict_type.lower() == "spanextraction":
                logger.info("Bert model enabled for span predictions")
                if args.model.lower() != "bert":
                    logger.warn(
                        f"Correct span extraction model from {args.model} to bert")
                # TODO: add more variants span extraction pre-trained model
                config = BertConfig.from_pretrained("bert-base-uncased")
                model = SpanPredictor.from_pretrained(
                    "bert-base-uncased", config=config)
            if args.device == "cuda" and torch.cuda.device_count() > 1:
                if args.n_gpu == 1:
                    logger.warning("User specified one gpu but there are actually {}, it has been corrected".format(
                        torch.cuda.device_count()))
                    args.n_gpu = torch.cuda.device_count()
                model = torch.nn.DataParallel(model)

        model.to(torch.device(args.device))

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
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')

        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key): value for key, value in state_dict.items()}
        model = MyBart.from_pretrained("bart-large",
                                       state_dict=convert_to_single_gpu(torch.load(checkpoint)))
        logger.info("Loading checkpoint from {}".format(checkpoint))
        # if args.single_gpu:
        model.to(torch.device(args.device))
        # elif torch.cuda.is_available():
        #     model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_data, args.predict_type,
                        device=args.device, save_predictions=True)
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

            if args.eval_period == 0 or global_step % args.eval_period == 0:
                model.eval()
                # import pdb
                # pdb.set_trace()

                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_data,
                                    args.predict_type, device=args.device, save_predictions=True)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em*100,
                    epoch))
                epoch_ems[epoch] = str(curr_em*100)
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k: v.cpu()
                                        for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(
                        args.output_dir, "best-model.pt"))
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


def inference(model, dev_data, predict_type, device="cuda", save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id

    # if predict_type == "thresholding":
    #     # generate answer
    #
    # elif predict_type == "SpanSeqGen":
    #     print("not implemented yet")
    #     exit()
    if predict_type.lower() == "spanseqgen":
        for i, batch in enumerate(dev_data.dataloader):
            
            if torch.cuda.is_available():
                batch = [b.to(device) for b in batch]

            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=dev_data.args.num_beams,
                                     max_length=dev_data.args.max_output_length,
                                     early_stopping=True,)
            # Q: span extraction: what it generates?
            # overwrite bert generate function

            # TODO: if it's logits then use decode function in span utils
            for input_, output in zip(batch[0], outputs):
                pred = dev_data.decode(output)
                predictions.append(pred)
    elif predict_type.lower() == "spanextraction":
        # import pdb
        # pdb.set_trace()
        # print("enable interactive mode and test code of evaluating span extraction")
        for i, batch in enumerate(dev_data.dataloader):
            if torch.cuda.is_available():
                batch = [b.to(device) for b in batch]
            print("inference span in batch ", i)
            start_logits, end_logits = model(input_ids=batch[0], attention_mask=batch[1],
                                             token_type_ids=batch[2], inputs_embeds=None,
                                             start_positions=batch[3], end_positions=batch[4], answer_mask=batch[5],
                                             is_training=False)
            # Q: span extraction: what it generates?
            # overwrite bert generate function

            # TODO: add these arguments later
            top_k_answers = 5
            max_answer_length = 10
            text_predictions = decode(start_logits, end_logits, batch[0],
                                      dev_data.tokenizer, top_k_answers, max_answer_length)

            # TODO: if it's logits then use decode function in span utils
            # import pdb
            # pdb.set_trace()
            # print("test text predictions")
            for input_, pred in zip(batch[0], text_predictions):
                # import pdb
                # pdb.set_trace()
                # print("expect pred to be a list of tokens")
                # pred = dev_data.decode(output)
                predictions.append(pred)
            print("predictions size / input size ",
                  len(predictions), "/ ", len(batch[0])*(i+1))
            if len(predictions) != len(batch[0])*(i+1):
                import pdb
                pdb.set_trace()
    # NOTE: in the decoding module, the input batch size matches the predictions size 
    # therefore, it could be something wrong with the dataloader
    # import pdb
    # pdb.set_trace()
    # print("check dev data length")
    # predictions here are all dev data's predictions
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))
