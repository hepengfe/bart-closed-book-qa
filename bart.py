from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_bart import shift_tokens_right
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple
class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        """
        Return loss for training mode and return outputs for evaluation mode
        :param input_ids:
        :param attention_mask:
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param use_cache:
        :param is_training:
        :return:
        """
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

class MyBart2(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
                use_cache=False, is_training=False):
        """
        Return loss for training mode and return outputs for evaluation mode
        :param input_ids:
        :param attention_mask:
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param use_cache:
        :param is_training:
        :return:
        """
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids
        def cp_forward(
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs: Optional[Tuple] = None,
                decoder_attention_mask=None,
                decoder_cached_states=None,
                use_cache=False,
        ):
            # From BartModel  (super()) ->
            # make masks if user doesn't supply
            if not use_cache:
                decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                    self.config,
                    input_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_padding_mask=decoder_attention_mask,
                    causal_mask_dtype=super(MyBart2, self).shared.weight.dtype,
                )
            else:
                decoder_padding_mask, causal_mask = None, None

            assert decoder_input_ids is not None
            if encoder_outputs is None:
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            assert isinstance(encoder_outputs, tuple)
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                decoder_input_ids,
                encoder_outputs[0],
                attention_mask,
                decoder_padding_mask,
                decoder_causal_mask=causal_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
            )
            # Attention and hidden_states will be [] or None if they aren't needed
            decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
            assert isinstance(decoder_outputs[0], torch.Tensor)
            encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
            return decoder_outputs + encoder_outputs
        outputs = cp_forward(
            input_ids = input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )















        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                            decoder_input_ids.view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]

# class MyBartForConditionalGeneration(BartForConditionalGeneration):
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
#             config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
#             (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
#
#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if labels is not None:
#             if decoder_input_ids is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )