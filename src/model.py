import numpy as np
import random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

N_HIDDEN_LAYERS = 2
P_DROPOUT = 0.1

class TransformBlock(nn.Module):
    def __init__(self, hinp, factor, p_dropout=0.1):
        transform_hidden_size = int(hinp*factor)
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hinp, transform_hidden_size),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(transform_hidden_size, hinp),
        )

    def forward(self, x):
        return x + self.block(x)


class GPT2TransformedLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, n_blocks=2, factor=0.5, p_dropout=0.1, is_train=True):
        super().__init__(config)
        self.is_train = is_train #if not training, output has to be a little different to be compatible with huggingface geneerate pipeline
        hidden_size = self.transformer.config.hidden_size
        transform_heads = [TransformBlock(hidden_size, factor, p_dropout)
                                for _ in range(n_blocks)]
        self.transform_heads = nn.ModuleList(transform_heads)                                

    def freeze_except_transform_head(self):
        for param in self.parameters():
            param.requires_grad = False
        for head in self.transform_heads:
            for param in head.parameters():
                param.requires_grad = True 

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # lm_logits = self.lm_head(hidden_states)
        for transform_head in self.transform_heads:
            hidden_states = transform_head(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if self.is_train:
            ret_hidden_states=(transformer_outputs.hidden_states, hidden_states)
        else:
            ret_hidden_states=hidden_states

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=ret_hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )