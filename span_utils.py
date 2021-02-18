
from tqdm import tqdm
def preprocess(tokenizer, questions, answers, metadata, all_titles, all_passages,
               is_training, max_input_length, max_n_answers):
    '''
    tokenizer: bertTokenizer
    questions: a list of strings
    answers: a list of lists of strings (note: OK for NQ, but need modification for AmbigQA)
    all_titles: a list of lists of k strings
    all_passages: a list of liss of k strings
    '''
    assert len(questions)==len(all_titles)==len(all_passages)==len(metadata)

    inputs = []
    # import pdb
    # pdb.set_trace()

    print("Start concatenate question and passages")
    for question, titles, passages in tqdm(zip(questions, all_titles, all_passages)):

        concatenated_context = ""
        for title, passage in zip(titles, passages):
            if len(concatenated_context)>0:
                concatenated_context += " [SEP] "
            concatenated_context += title + " [SEP] " + passage
        inputs.append((question, concatenated_context))

    # encoding is time consuming part, this version of transformer doesn't have BartTokenizerFast
    input_data = tokenizer.batch_encode_plus(inputs, padding="max_length", max_length=max_input_length,
                                             truncation=True, return_attention_mask = True, return_token_type_ids = True)

    answer_data = tokenizer.batch_encode_plus(answers)

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    token_type_ids = input_data["token_type_ids"]
    # import pdb
    # pdb.set_trace()


    # as some of input ids will be skipped 
    new_input_ids = []
    new_attention_mask = []
    new_token_type_ids = []

    start_positions = []
    end_positions = []
    answer_mask = []

    for curr_input_ids, curr_attention_mask, curr_token_type_ids, curr_metadata in zip(
                input_ids, attention_mask, token_type_ids, metadata):

        # offset record sep token location in original text
        # Q: why it returns only one index? 
        # A: [...].index(<SEP> id)
        offset = 1 + curr_input_ids.index(tokenizer.sep_token_id)
        # start: curr_metadata[0]
        # end:  curr_metadata[1]
        # answer_data is batch encoding
        # answer_data._encoding is None


        # NOTE: [1:-1] is to slice out <SEP> token

        answer_input_ids = [answer_data["input_ids"][i][1:-1] for i in range(curr_metadata[0], curr_metadata[1])]

        # now, detect answer spans from passages
        detected_spans = []

        # answer_input_ids is a list of acceptable input ids
        for curr_answer_input_ids in answer_input_ids:
            for i in range(offset, len(curr_input_ids)-len(curr_answer_input_ids)+1): # scan through input ids (question and passages)
                if curr_input_ids[i:i+len(curr_answer_input_ids)]==curr_answer_input_ids:
                    detected_spans.append( (i, i+len(curr_input_ids)-1))
                    if len(detected_spans)==max_n_answers:
                        break

        if is_training and len(detected_spans)==0:
            continue
        # start_positions =  [s[0], s[0], 0, 0]

        # TODO: it could have some better way to save RAM but current implementation ensures correctness
        new_input_ids.append(curr_input_ids)
        new_attention_mask.append(curr_attention_mask)
        new_token_type_ids.append(curr_token_type_ids) 

        # 
        


        start_positions.append([s[0] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        end_positions.append([s[1] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        answer_mask.append([1 for _ in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
    answer_coverage_rate = len(new_input_ids)/len(input_ids) # measure how often answers appear in passages
    print("answer coverage rate by passages: ",
          answer_coverage_rate, "    ", len(new_input_ids), "/", len(input_ids))
    return {"input_ids": new_input_ids, "attention_mask": new_attention_mask, "token_type_ids": new_token_type_ids,
            "start_positions": start_positions, "end_positions": end_positions, "answer_mask": answer_mask, "answer_coverage_rate": answer_coverage_rate}

def decode(start_logits, end_logits, input_ids, tokenizer, top_k_answers, max_answer_length):
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

    all_predictions = []


    for curr_start_logits, curr_end_logits, curr_input_ids in \
            zip(start_logits, end_logits, input_ids):

        assert len(curr_start_logits)==len(curr_end_logits)==len(curr_input_ids)
        # import pdb
        # pdb.set_trace()
        # print("check curr_input_ids and it should be a list")
        offset = 1 + curr_input_ids.tolist().index(tokenizer.sep_token_id)

        curr_start_logits = curr_start_logits[offset:]
        curr_end_logits = curr_end_logits[offset:]
        scores = []
        for (i, s) in enumerate(curr_start_logits):
            for (j, e) in enumerate(curr_end_logits[i:i+max_answer_length]):
                scores.append(((i, i+j), s+e))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        nbest = []
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

            answer_text = tokenizer.decode(
                input_ids[offset+start_index:offset+end_index+1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).strip()

            nbest.append({
                'text': answer_text,
                'log_softmax': score})

            if len(nbest)==top_k_answers:
                break
        # [{'text': '', 'log_softmax': tensor(1.1091, device='cuda:0', grad_fn=<AddBackward0>)} ]
        # a list of dictionary
        all_predictions.append(nbest)
    # outputs a dictionary of 
    # import pdb
    text_predictions = []
    for preds in all_predictions:
        text_predictions.append([ pred['text'] for pred in preds ])
    return text_predictions
