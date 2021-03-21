from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right, BartEncoder, BartEncoderLayer, BartModel # _prepare_bart_decoder_inputs,
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple




def invert_mask(attention_mask):
    """Not avaiable in huggingface 4.3.3. So I pasted it here. 

    Args:
        attention_mask ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

class MyBart(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.is_ambig = False
    def set_ambig(self):
        """
        Set ambig QA type there is no good way passing the variable.
        """
        self.is_ambig = False
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None, 
                past_key_values=None, head_mask = None,  return_dict=None, output_attentions=None, output_hidden_states=None,
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
            # don't know why but merge the code from transfromer 2.9
            # index_of_eos = (input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
            # decoder_start_token_id = decoder_input_ids[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
            # _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, decoder_start_token_id)
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        else:
            _decoder_input_ids = decoder_input_ids
        # if past_key_values is not None:
        #     decoder_input_ids = None
        outputs = super(MyBart, self).forward(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            # decoder_cached_states=decoder_cached_states,   # no longer in 4.3.3
            past_key_values=past_key_values, 
            head_mask = head_mask, 
            labels = decoder_input_ids,  # NOTE: it might causes bug which return different value 
            return_dict=True,
            use_cache=use_cache, 
        )
        # outputs = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     encoder_outputs=encoder_outputs,
        #     decoder_input_ids=_decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     # decoder_cached_states=decoder_cached_states,   # no longer in 4.3.3
        #     past_key_values=past_key_values, 
        #     head_mask = head_mask, 
        #     return_dict=return_dict,
        #     use_cache=use_cache,
        # )

        # NOTE: not sure if it's the same as function
        # lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
            # loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                            # decoder_input_ids.view(-1))
            return outputs.loss
        return outputs



# implement an encoder layer with hidden states as input arguments


class CPBartEncoder(BartEncoder):
    # BertEncoder
    # def __init__(self, config):
    #     super().__init__()
    #     self.config = config
    #     self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    #
    # def forward(
    #         self,
    #         hidden_states,
    #         attention_mask=None,
    #         head_mask=None,
    #         encoder_hidden_states=None,
    #         encoder_attention_mask=None,
    #         past_key_values=None,
    #         use_cache=None,
    #         output_attentions=False,
    #         output_hidden_states=False,
    #         return_dict=True,
    # ):



    # BartEncoder
    # class BartEncoder(nn.Module):
    #     """
    #     Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    #     is a :class:`BartEncoderLayer`.
    #
    #     Args:
    #         config: BartConfig
    #     """
    #
    #     def __init__(self, config: BartConfig, embed_tokens):
    #         super().__init__()
    #
    #         self.dropout = config.dropout
    #         self.layerdrop = config.encoder_layerdrop
    #         self.output_attentions = config.output_attentions
    #         self.output_hidden_states = config.output_hidden_states
    #
    #         embed_dim = embed_tokens.embedding_dim
    #         self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
    #         self.padding_idx = embed_tokens.padding_idx
    #         self.max_source_positions = config.max_position_embeddings
    #
    #         self.embed_tokens = embed_tokens
    #         if config.static_position_embeddings:
    #             self.embed_positions = SinusoidalPositionalEmbedding(
    #                 config.max_position_embeddings, embed_dim, self.padding_idx
    #             )
    #         else:
    #             self.embed_positions = LearnedPositionalEmbedding(
    #                 config.max_position_embeddings, embed_dim, self.padding_idx,
    #             )
    #         self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
    #         self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
    #         # mbart has one extra layer_norm
    #         self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None
    #
    #     def forward(
    #         self, input_ids, attention_mask=None,
    #     ):

    def __init__(self, config, embed_tokens, gradient_cp = True):
        super().__init__(config, embed_tokens)
        gradient_cp = False
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
                    x, attn = torch.utils.checkpoint.checkpoint(create_custom_encoder_forward(encoder_layer), x, attention_mask)
                else:
                    # import pdb
                    # pdb.set_trace()
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


        # reinitialize cutomized bartencoder if gradient checkpoint is enabled
        if self.gradient_cp:
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

        if not self.gradient_cp:  # if gradient checkpoint is not enabled, use the default checkpoint method
            return super(MyBartModel, self).forward(input_ids, \
                                             attention_mask,\
                                             decoder_input_ids, \
                                             encoder_outputs, \
                                             decoder_attention_mask,\
                                              decoder_cached_states,\
                                              use_cache)
        else:
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

            def create_custom_encoder_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            if encoder_outputs is None:
                # import pdb
                # pdb.set_trace()
                # encoder_outputs  = torch.utils.checkpoint.checkpoint(create_custom_encoder_forward(self.encoder), input_ids, attention_mask)
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
