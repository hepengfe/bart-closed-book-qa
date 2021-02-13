

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

    for question, titles, passages in zip(questions, all_titles, all_passages):

        concatenated_context = ""
        for title, passage in zip(titles, passages):
            if len(concatenated_context)>0:
                concatenated_context += " [SEP] "
            concatenated_context += title + " [SEP] " + passage
        inputs.append((question, concatenated_context))

    input_data = tokenizer.batch_encode_plus(inputs, padding="max_length", max_length=max_input_length,
                                             return_offsets_mapping=True, truncation=True)
    answer_data = tokenizer.batch_encode_plus(answers)

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    token_type_ids = input_data["token_type_ids"]
    offset_mapping = input_data["offset_mapping"]

    start_positions = []
    end_positions = []
    answer_mask = []
    for curr_input_ids, curr_attention_mask, curr_token_type_ids, curr_offset_mapping, curr_metadata in zip(
                input_ids, attention_mask, token_type_ids, curr_offset_mapping, metadata):

        offset = 1 + curr_offset_mapping[1:].index((0, 0)) # start of the passages
        answer_input_ids = [answer_data[i][1:-1] for i in range(curr_metadata[0], curr_metadata[1])]

        # now, detect answer spans from passages
        detected_spans = []
        for curr_answer_input_ids in answer_input_ids:
            for i in range(offset, len(curr_input_ids)-len(curr_answer_input_ids)+1):
                if curr_input_ids[i:i+len(curr_answer_input_ids)]==curr_answer_input_ids:
                    detected_spans.append(i, i+len(curr_input_ids)-1)
                    if len(detected_spans)==max_n_answers:
                        break

        if is_training and len(detected_spans)==0:
            continue

        start_positions.append([s[0] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        end_positions.append([s[1] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        answer_mask.append([1 for _ in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])


    # offset_mapping and raw_inputs are needed for span decoding. Others are needed to be fed into
    # the model
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
            "start_positions": start_positions, "end_positions": end_positions, "answer_mask": answer_mask,
            "offset_mapping": offset_mapping, "raw_inputs": inputs}


def decode(start_logits, end_logits, offset_mapping, raw_inputs,
           top_k_answers, max_answer_length):

    assert len(start_logits)==len(end_logits)==len(offset_mapping)==len(raw_inputs)

    all_predictions = []

    for curr_start_logits, curr_end_logits, curr_offset_mapping, curr_raw_inputs in \
            zip(start_logits, end_logits, offset_mapping, raw_inputs):

        assert len(curr_start_logits)==len(curr_end_logits)==len(curr_offset_mapping)
        offset = 1 + curr_offset_mapping[1:].index((0, 0))

        curr_start_logits = curr_start_logits[offset:]
        curr_end_logits = curr_end_logits[offset:]

        for (i, s) in enumerate(curr_start_logits):
            for (j, e) in enumerate(curr_end_logits[i:i+max_answer_length]):
                scores.append(((i, i+j), s+e))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        nbext = []
        for (start_index, end_index), score in scores:
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            if any([start_index<=prev_start_index<=prev_end_index<=end_index or
                    prev_start_index<=start_index<=end_index<=prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals]):
                continue

            char_start, char_end = curr_offset_mapping[s+offset][0], curr_offset_mapping[e+1+offset][1]
            answer_text = curr_raw_inputs[char_start:char_end]

            nbest.append({
                'text': answer_text,
                'log_softmax': score})

            if len(nbest)==top_k_answers:
                break
        all_predictions.append(nbest)

    return all_predictions
