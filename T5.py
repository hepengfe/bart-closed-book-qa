from transformers import T5ForConditionalGeneration

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_bart import shift_tokens_right
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss



class MyT5(T5ForConditionalGeneration):

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        return self.forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, encoder_outputs)

        # def forward(
        #         self,
        #         input_ids=None,
        #         attention_mask=None,
        #         decoder_input_ids=None,
        #         decoder_attention_mask=None,
        #         encoder_outputs=None,
        #         past_key_values=None,
        #         head_mask=None,
        #         inputs_embeds=None,
        #         decoder_inputs_embeds=None,
        #         labels=None,
        #         use_cache=None,
        #         output_attentions=None,
        #         output_hidden_states=None,
        #         return_dict=None,
        # ):
    #
    #     if is_training:
    #         _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
    #     else:
    #         _decoder_input_ids = decoder_input_ids
    #
    #     encoder_outputs = self.encoder()
    #
    #     outputs = self.model(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         decoder_input_ids=_decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         decoder_cached_states=decoder_cached_states,
    #         use_cache=use_cache,
    #     )
    #     lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
    #     if is_training:
    #         loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
    #         loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
    #                           decoder_input_ids.view(-1))
    #         return loss
    #     return (lm_logits, ) + outputs[1:]