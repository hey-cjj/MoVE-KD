import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from transformers.models.clip.modeling_clip import CLIPEncoderLayer,CLIPEncoder,CLIPVisionTransformer
from typing import Any, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput,BaseModelOutputWithPooling
from dataclasses import dataclass

@dataclass
class LoRAMOEBaseModelOutputWithPooling(BaseModelOutputWithPooling):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    routings: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LoRAMOEBaseModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    routings: Optional[Tuple[torch.FloatTensor]] = None

class LoRALayer(nn.Module):
    def __init__(self, in_channel, out_channel, rank=32, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        self.lora_A = nn.Linear(in_channel, rank)
        self.lora_B = nn.Linear (rank, out_channel)
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else 0
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lora_B.weight, nonlinearity='relu')
        if self.lora_A.bias is not None:
            nn.init.zeros_(self.lora_A.bias)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, X):             
        result = (self.lora_B(self.lora_A(self.lora_dropout(X)))) * self.scaling
        return result
    
class MoE_LoRA_CLIP(nn.Module):
    def __init__(self,args,lora_rank: int,lora_alpha: int,num_experts: int,
                 dense_moe =False, original_module: nn.Module = None):
        super().__init__()
        self.args = args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.original_module = original_module
        self.dense_moe = dense_moe

        in_features = original_module.fc1.in_features
        out_features = original_module.fc1.out_features

        self.moe_down = nn.ModuleList()
        self.moe_up = nn.ModuleList()
        self.original_module.fc1.weight.requires_grad = False
        self.original_module.fc2.weight.requires_grad = False

        for _ in range(num_experts):
            self.moe_down.append(LoRALayer(
                in_channel=in_features,
                out_channel=out_features,
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
            self.moe_up.append(LoRALayer(
                in_channel=out_features,
                out_channel=in_features,
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
        
        self.router = nn.Linear(in_features, self.num_experts)

    def forward_lora_moe(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out_per_expert = []
        for i in range(self.num_experts):
            lora_out_per_expert.append(moe[i](x))

        lora_out = torch.stack(lora_out_per_expert, 2)

        lora_out = (lora_out * routing[:,:,:,None]).sum(2)

        moe_out = original_out + lora_out
        return moe_out

    def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
        original_out = original_proj(x)

        lora_out = torch.zeros_like(original_out)
        for i in range(self.num_experts):
            id1, id2, _ = torch.where(routing_idx==i)
            lora_out[id1, id2] = moe[i](x[id1, id2])

        moe_out = original_out + lora_out
        return moe_out


    def forward(self, x):
        logits = self.router(x)
        routing = F.softmax(logits, dim=-1)
        index = routing.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        expert_choice = y_hard - routing.detach() + routing

        if self.dense_moe:
            down_moe_out = self.forward_lora_moe(x, self.original_module.fc1, routing, self.moe_down)
        else:
            down_moe_out = self.forward_lora_moe_sparse(x, self.original_module.fc1, index, self.moe_down)

        x = self.original_module.activation_fn(down_moe_out)
        
        if self.dense_moe:
            x = self.forward_lora_moe(x, self.original_module.fc2, routing, self.moe_up)
        else:
            x = self.forward_lora_moe_sparse(x, self.original_module.fc2, index, self.moe_up)
        return x, (routing, expert_choice)

class MoECLIPEncoderLayer(CLIPEncoderLayer):
    def forward( self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # MoE_LoRA_CLIP output: hidden_states, (routing, expert_choice)
        moe_routing = None
        if isinstance(hidden_states, tuple):   
            hidden_states, moe_routing = hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        
        outputs += (moe_routing,)

        return outputs

class MoECLIPEncoder(CLIPEncoder):
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LoRAMOEBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_moe_routings = ()

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         encoder_layer.__call__,
            #         hidden_states,
            #         attention_mask,
            #         causal_attention_mask,
            #         output_attentions,
            #     )
            # else:
            #     layer_outputs = encoder_layer(
            #         hidden_states,
            #         attention_mask,
            #         causal_attention_mask,
            #         output_attentions=output_attentions,
            #     )
            
            layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            all_moe_routings += (layer_outputs[2 if output_attentions else 1], )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, all_moe_routings] if v is not None)
        return LoRAMOEBaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=encoder_states, 
            attentions=all_attentions,
            routings=all_moe_routings,
        )

class MoECLIPVisionTransformer(CLIPVisionTransformer):
     def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LoRAMOEBaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return LoRAMOEBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            routings= encoder_outputs.routings
        )

