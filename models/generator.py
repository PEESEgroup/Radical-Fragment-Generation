from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn

class BDEConditionedGPT(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

        self.bde_projector = nn.Linear(1, config.n_embd * 2)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        bde=None,
        labels=None,
        inputs_embeds=None,
        **kwargs
    ):
        inputs_embeds = self.transformer.wte(input_ids)

        if bde is not None:
            # Get gamma and beta
            bde_params = self.bde_projector(bde.unsqueeze(-1)) 
            gamma, beta = bde_params.chunk(2, dim=-1)

            # FiLM
            inputs_embeds = (1 + gamma.unsqueeze(1)) * inputs_embeds + beta.unsqueeze(1)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )