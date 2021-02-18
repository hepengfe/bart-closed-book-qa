import torch
import  numpy
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForQuestionAnswering, AlbertForQuestionAnswering

from span_utils import decode


class SpanPredictor(BertForQuestionAnswering):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None, answer_mask=None,
                is_training=False):
        # check input embeds

        # input_ids, attention_mask, token_type_ids should have a shape of [batch_size, input_length]
        # start_positions, end_positions, answer_mask should have a shape of [batch_size, max_n_answers]
        output = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           inputs_embeds=inputs_embeds)[0]
        # output will have a shape of [batch_size, input_length, hidden_size]

        logits = self.qa_outputs(output) # [batch_size, input_length, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # [batch_size, input_length]
        end_logits = end_logits.squeeze(-1) # [batch_size, input_length]

        if is_training:
            return get_loss(start_positions, end_positions, answer_mask, start_logits, end_logits)
        else:
            return start_logits, end_logits

    def generate(self, input_ids=None, attention_mask=None,
                 token_type_ids=None, inputs_embeds=None,
                 start_positions=None, end_positions=None, answer_mask=None,
                 is_training=False):
        start_logits, end_logits = self.forward(input_ids=None, attention_mask=None,
                                                token_type_ids=None, inputs_embeds=None,
                                                start_positions=None, end_positions=None, answer_mask=None,
                                                is_training=False)
        predictions = decode(start_logits, end_logits, input_ids,
                              tokenizer, top_k_answers, max_answer_length)

        return predictions

def get_loss(start_positions, end_positions, answer_mask, start_logits, end_logits):
    answer_mask = answer_mask.type(torch.FloatTensor).cuda()
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # NOTE: torch unbind serves a way to generate a list of tensors
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                    for (_start_positions, _span_mask) \
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                    for (_end_positions, _span_mask) \
                    in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
        torch.cat([t.unsqueeze(1) for t in end_losses], dim=1) # sum up start loss and end loss for each prediction
    # N: batch size
    # M: output length (or input?) it should be input as 
    N = len(start_positions) 
    M = len(start_positions[0])
    loss_tensor=loss_tensor.view(N, M, -1).max(dim=1)[0] # taking the maximum loss along dimension of input
    span_loss = _take_mml(loss_tensor)
    return span_loss

def _take_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
            - loss_tensor - 1e10 * (loss_tensor==0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood + \
                                torch.ones(loss_tensor.size(0)).cuda()*(marginal_likelihood==0).float()))

