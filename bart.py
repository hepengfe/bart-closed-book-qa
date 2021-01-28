from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_bart import shift_tokens_right, _prepare_bart_decoder_inputs, BartEncoder, EncoderLayer, BartModel, invert_mask
import random
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



class CPBartEncoder(BartEncoder):
    def __init__(self, config, embed_tokens, gradient_cp = True):
        super().__init__(config, embed_tokens)
        self.gradient_cp = gradient_cp


    def forward(self, input_ids, attention_mask=None,):
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []

        def create_custom_encoder_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for i, encoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                if self.gradient_cp:
                    import pdb
                    pdb.set_trace()
                    x, attn = torch.utils.checkpoint.checkpoint(create_custom_encoder_forward(encoder_layer), x, attention_mask[i] )
                else:
                    x, attn = encoder_layer(x, attention_mask)
            if self.output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions


class MyBartModel(BartModel):
    def __init__(self, config, gradient_cp):
        super(MyBartModel, self).__init__(config)
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        #
        # padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)


        # self.decoder = BartDecoder(config, self.shared)
        self.gradient_cp = gradient_cp
        self.encoder = CPBartEncoder(config, self.shared, gradient_cp)
        self.init_weights()


    def set_gradient_cp(self, gradient_cp):
        """
        If gradient_cp is true, reinitialize encoder and decoder and their weights
        :param gradient_cp:
        :return:
        """
        self.gradient_cp = gradient_cp
        self.encoder = CPBartEncoder(config, self.shared, gradient_cp)
        self.init_weights()
    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=False,
    ):

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
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

class MyBartForCondGen(BartForConditionalGeneration):
    def __init__(self, config):
        super(MyBartForCondGen, self).__init__(config)
        self.model = MyBartModel(config, gradient_cp=True) # reintialize base model

    #     self.model.set_gradient_cp(True)
    #
    #
    # def set_gradient_cp(self, gradient_cp):
    #     self.model.set_gradient_cp(gradient_cp)







    # def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
    #             decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
    #             use_cache=False, is_training=False):
    #     """
    #     Return loss for training mode and return outputs for evaluation mode
    #     :param input_ids:
    #     :param attention_mask:
    #     :param encoder_outputs:
    #     :param decoder_input_ids:
    #     :param decoder_attention_mask:
    #     :param decoder_cached_states:
    #     :param use_cache:
    #     :param is_training:
    #     :return:
    #     """
    #     if is_training:
    #         _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
    #     else:
    #         _decoder_input_ids = decoder_input_ids
    #
    #     def cp_bart_forward(
    #             input_ids,
    #             attention_mask=None,
    #             decoder_input_ids=None,
    #             encoder_outputs: Optional[Tuple] = None,
    #             decoder_attention_mask=None,
    #             decoder_cached_states=None,
    #             use_cache=False,
    #     ):
    #         # From BartModel  (super()) ->
    #         # make masks if user doesn't supply
    #         if not use_cache:
    #             decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
    #                 self.config,
    #                 input_ids,
    #                 decoder_input_ids=decoder_input_ids,
    #                 decoder_padding_mask=decoder_attention_mask,
    #                 causal_mask_dtype=super(MyBartForCondGen, self).shared.weight.dtype,
    #             )
    #         else:
    #             decoder_padding_mask, causal_mask = None, None
    #
    #         assert decoder_input_ids is not None
    #         if encoder_outputs is None:
    #             encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    #         assert isinstance(encoder_outputs, tuple)
    #         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    #         decoder_outputs = self.decoder(
    #             decoder_input_ids,
    #             encoder_outputs[0],
    #             attention_mask,
    #             decoder_padding_mask,
    #             decoder_causal_mask=causal_mask,
    #             decoder_cached_states=decoder_cached_states,
    #             use_cache=use_cache,
    #         )
    #         # Attention and hidden_states will be [] or None if they aren't needed
    #         decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
    #         assert isinstance(decoder_outputs[0], torch.Tensor)
    #         encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
    #         return decoder_outputs + encoder_outputs
    #     # want to pass status
    #     outputs = cp_bart_forward(
    #         input_ids = input_ids,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         decoder_input_ids=_decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         decoder_cached_states=decoder_cached_states,
    #         use_cache=use_cache,
    #     )















        # lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        # if is_training:
        #     loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
        #     loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
        #                     decoder_input_ids.view(-1))
        #     return loss
        # return (lm_logits, ) + outputs[1:]

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