
import pdb
from re import A
from IPython import embed
from tqdm import tqdm
import numpy as np
import json
import os
import pickle # pickle is faster than json if the user doesn't need readability
def preprocess(tokenizer, questions, answers, metadata, all_titles, all_passages,
               is_training, max_input_length, max_n_answers, encoded_path):
    '''
    Genereate a list of passage containing the answer.

    tokenizer: bertTokenizer
    questions: a list of strings
    answers: a list of lists of strings (note: OK for NQ, but need modification for AmbigQA)
    metadata: start of the answer in the flattened list: curr_metadata[0] / end of the answer in the flattened list:  curr_metadata[1]
    all_titles: a list of lists of k strings, each question has a list of string titles
    all_passages: a list of lists of k strings
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

    
    contained = []
    for input, (s, e) in zip(inputs, metadata):
        curr_answers = answers[s:e]
        contained.append(any([answer.lower() in input[1].lower()
                            for answer in curr_answers]))
    print(np.mean(contained))
    
    encoded_input_path = encoded_path.replace(".json", "_input.p") # rewrite the json file to pickle file path
    encoded_answer_path = encoded_path.replace(".json", "_answer.p")

    if os.path.exists(encoded_input_path) and os.path.exists(encoded_answer_path):
        with open(encoded_input_path, "rb") as fp:
            input_data = pickle.load(fp)
        with open(encoded_answer_path, "rb") as fp:
            answer_data = pickle.load(fp)
        # input_data = json.load(encoded_input_path)
        # answer_data = json.load(encoded_answer_path)
        # input_data = pickle.load()
        
    else:  # it also handles special case that there is no 
        print("Not found encoded cache, now encoding QP concatenation ")
        # encoding is time consuming part, this version of transformer doesn't have BartTokenizerFast
        input_data = tokenizer.batch_encode_plus(inputs, padding="max_length", max_length=max_input_length,
                                                truncation=True, return_attention_mask = True, return_token_type_ids = True)
        answer_data = tokenizer.batch_encode_plus(answers)
        with open(encoded_input_path, "wb") as fp:
            # json.dump(input_data, fp)
            pickle.dump(input_data, fp)
        with open(encoded_answer_path, "wb") as fp:
            # json.dump(answer_data, fp)
            pickle.dump(answer_data, fp)
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
                    detected_spans.append( (i, i+len(curr_input_ids)-1))
                    if len(detected_spans)==max_n_answers:
                        break
        
        if len(detected_spans) == 0:
            continue
        # TODO: uncomment the original code
        # if is_training and len(detected_spans)==0:
        #     continue


        # TODO: it could have some better way to save RAM but current implementation ensures correctness

        # NOTE: add into output data if there is detected spans
        new_input_ids.append(curr_input_ids)
        new_attention_mask.append(curr_attention_mask)
        new_token_type_ids.append(curr_token_type_ids) 
        start_positions.append([s[0] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        end_positions.append([s[1] for s in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])
        answer_mask.append([1 for _ in detected_spans] + [0 for _ in range(max_n_answers-len(detected_spans))])

    print("check answer coverage rate")


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
    input_ids = input_ids.tolist()
    # import pdb
    # pdb.set_trace()
    print("speed test breakpoint 1")
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
        # import pdb
        # pdb.set_trace()
        # print("sorting")
        scores = [(score[0], score[1].item()) for score in scores]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # print("sorting finished")
        chosen_span_intervals = []
        nbest = [] # record n best for current question
        
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
            # import pdb
            # pdb.set_trace()
            
            # input ids = batch sz x seq length
            # NOTE: I think the bug is because inputs have two answers. Just something causes it has different shape.
            # import pdb
            # pdb.set_trace()
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
        # a list of dictionary
        all_predictions.append(nbest)
    # pdb.set_trace()
    print("speed test breakpoint 2")

    # import pdb
    text_predictions = []

    if len(all_predictions) == len(input_ids):
        pass
    else:
        import pdb
        pdb.set_trace()

    for preds in all_predictions:
        text_predictions.append([ pred['text'] for pred in preds ])
    return text_predictions
