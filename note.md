* cli.py   configure log and arguments
* run.py  training scheme
* bart.py   model
    - shift_tokens_right  \<sos>, 1, 2, \<eos> -> \<eos>, \<sos>, 1, 2
        * used during training, compare [2:]
    - inherit BartModel and add customized method
        * MyBart (custom class)
        * BartForConditionalGeneration (parent 1)
            - foward() what Sewon overwrites
            - self.model = BartModel(config)
                * forward method that we want to overwrite
        * BartPretrainedModel (parent 2)
    - Q: why customize forward function?
        * differentiate training/eval mode
        * pipeline = classifier_fc(pre_trained_model(input))
    - Q: why F.linear()?
        * such a function method is useful when we inherit model and do not want to write a new model initialization method

* T5.py
    - T5 model, encoder and decoder
    - T5 model architecture
        * T5PreTrainedModel
            - T5ForConditionalGeneration 
                - encoder = T5stack(encoder_block)
                - decoder = T5stack(decoder_block)  -> where 
* walkthrough the tutorial
    * preprocessing data: token -> id (tokenizer)
    * training and fine-tuning 
        * scheduler  (schedule learning rate)
        * trainer  (simply what we often wrote most, but it's not used in Sewon's code)
            - `TrainingArguments(...)`
            - `Trainer(...)`
* arguments/variables
    * transformer model has encoder and decoder
    * decoder_input_ids? is it for teacher forcing?
    * cache:
        - Without using cached hidden states : every step, the next token is predicted, but also all previous tokens are re-computed (which is useless because we already predicted it !)
        - Using cached hidden states : every step, the next token is predicted, but previous tokens are not re-computed, because we are using their cached states.
    * return dict
        - similar to return stuff depending on the train/eval mode
    * Seq2SeqLMOutput

* Recommended edit for checkpoint
    - checkpoint expensive computation or? I think so
    - Not checkpoint easy computed parts
    - BertEncoder already enabled checkpointing by argument gradient_checkpointing
    - There are encoder and decoder. I can try apply encoder and decoder
    - [bert with checkpointing](https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel)
* what is conditional generation?    
```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
hidden_states = layer_module(hidden_states, attention_mask)
hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)

        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

