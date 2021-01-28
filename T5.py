from transformers import T5ForConditionalGeneration

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_bart import shift_tokens_right
from transformers.modeling_t5 import T5PreTrainedModel, T5Stack
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

class T5StackCP(T5Stack):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_value_states=None,
            use_cache=False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        self.block.to("cuda") # TODO: add the device argument
        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            layer_head_mask = head_mask[i] if head_mask is not None else None
            # past_key_value_state = past_key_value_states[i] if past_key_values is not None else None

            # import pdb
            # pdb.set_trace()
            def create_custom_encoder_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            def create_custom_decoder_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            if not layer_module.is_decoder:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_encoder_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    past_key_value_state,
                    use_cache,
                )
            else:
                print("Decoder: ", layer_module.is_decoder)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_decoder_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    past_key_value_state,
                    use_cache,
                )


            #
            # layer_outputs = layer_module(
            #     hidden_states,
            #     attention_mask=extended_attention_mask,
            #     position_bias=position_bias,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_extended_attention_mask,
            #     encoder_decoder_position_bias=encoder_decoder_position_bias,
            #     head_mask=head_mask[i],
            #     past_key_value_state=past_key_value_state,
            #     use_cache=use_cache,
            # )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if self.output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)
        def cp_forward(
                input_ids=None,
                attention_mask=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_past_key_value_states=None,
                use_cache=True,
                lm_labels=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                head_mask=None,
        ):
            r"""
            lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                    Labels for computing the sequence classification/regression loss.
                    Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                    All labels set to ``-100`` are ignored (masked), the loss is only
                    computed for labels in ``[0, ..., config.vocab_size]``

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
                Classification loss (cross entropy).
            prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
                If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
            decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
                Contains pre-computed key and value hidden-states of the attention blocks.
                Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
                Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

        Examples::

            from transformers import T5Tokenizer, T5ForConditionalGeneration

            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
            input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
            loss, prediction_scores = outputs[:2]

            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
            input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
            outputs = model.generate(input_ids)
            """

            # Encode if needed (training, first prediction pass)
            if encoder_outputs is None:
                # Convert encoder inputs in embeddings if needed
                encoder_outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
                )

            hidden_states = encoder_outputs[0]

            if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids = self._shift_right(lm_labels)

            # If decoding with past key value states, only the last tokens
            # should be given as an input
            if decoder_past_key_value_states is not None:
                assert lm_labels is None, "Decoder should not use cached key value states when training."
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids[:, -1:]
                if decoder_inputs_embeds is not None:
                    decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_value_states=decoder_past_key_value_states,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
            )

            # insert decoder past at right place
            # to speed up decoding
            if use_cache is True:
                past = ((encoder_outputs, decoder_outputs[1]),)
                decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

            sequence_output = decoder_outputs[0]
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)

            decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
            if lm_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
                decoder_outputs = (loss,) + decoder_outputs

            return decoder_outputs + encoder_outputs
class MyT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        gradient_cp = True


        # overwrite default T5stack class if gradient_cp is True
        if gradient_cp:
            self.encoder = T5StackCP(self.config, self.shared)
            self.decoder = T5StackCP(self.config, self.shared)
            self.init_weights()

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
                use_cache=False, is_training=False):
        """
        Assume it's
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
        if check_point:

            outputs = cp_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_past_key_value_states=decoder_cached_states,
                use_cache=use_cache
            )
        else:
            # TODO: pending check
            outputs = super.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    encoder_outputs=encoder_outputs,
                                    decoder_input_ids=_decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    decoder_past_key_value_states=decoder_cached_states,
                                    use_cache=use_cache)

        # import pdb
        # pdb.set_trace()
        # lm_logits = F.linear(outputs[0], self.shared.weight.T) # remove bias=self.final_logits_bias from the argument
        lm_logits = outputs[0]
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                            decoder_input_ids.view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]
