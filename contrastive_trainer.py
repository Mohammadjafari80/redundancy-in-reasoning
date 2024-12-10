import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer

class SimpleInfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, pos, neg):
        """
        Args:
            query: [B, E] Tensor - anchor/query embeddings
            pos: [B, E] Tensor - positive embeddings (to be pulled closer to query)
            neg: [B, E] Tensor - negative embeddings (to be pushed away from query)
        Returns:
            Scalar InfoNCE loss.
        """
        # Normalize embeddings to unit vectors
        query = F.normalize(query, dim=-1)  # [B, E]
        pos = F.normalize(pos, dim=-1)      # [B, E]
        neg = F.normalize(neg, dim=-1)      # [B, E]

        # Positive similarity: Cosine similarity between query and corresponding positive
        pos_sim = torch.sum(query * pos, dim=-1)  # [B]

        # Negative similarity: Cosine similarity between query and corresponding negative
        neg_sim = torch.sum(query * neg, dim=-1)  # [B]

        # Combine similarities into logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)  # [B, 2]

        # Labels: First column (positive similarity) is the correct class
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)

        # Apply cross-entropy loss (temperature scaling)
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, contrastive_weight=0, lm_weight=1, temperature=0.5, tkn=None, completion_only=False, layers_to_use=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Contrastive loss parameters
        self.temperature = temperature
        self.info_nce = SimpleInfoNCE(temperature=temperature)

        # Weights for balancing losses
        if contrastive_weight < 0:
            raise ValueError("contrastive_weight must be >= 0")
        if lm_weight < 0:
            raise ValueError("lm_weight must be >= 0")

        self.contrastive_weight = contrastive_weight
        self.lm_weight = lm_weight  # Weight for language modeling loss
        self.tkn = tkn
        self.completion_only = completion_only

        # Layer-wise contrastive loss parameters
        if layers_to_use is None:
            self.layers_to_use = [-1]  # Default: only use the last layer
        else:
            self.layers_to_use = layers_to_use

    def encode(self, model, x, layer_idx=-1):
        # output_hidden_states must be enabled during model initialization
        out = model(**x, output_hidden_states=True).hidden_states[layer_idx][:, -1, :]
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get('text_input_ids')
        attention_mask = inputs.get('text_attention_mask')

        labels = None
        if self.completion_only:
            labels = inputs.get('solution_text_input_ids')  # Only consider the loss from response
        else:
            labels = input_ids.clone()  # Consider the loss from prompt and response

        if self.tkn is not None and self.tkn.pad_token_id is not None:
            labels[labels == self.tkn.pad_token_id] = -100

        # Compute standard language modeling loss
        lm_loss = 0
        if self.lm_weight > 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            lm_loss = outputs.loss

        # Compute contrastive loss
        contrastive_loss = 0
        if self.contrastive_weight > 0:
            query = {'input_ids': inputs.get('query_input_ids'),
                     'attention_mask': inputs.get('query_attention_mask')}
            pos = {'input_ids': inputs.get('pos_input_ids'),
                   'attention_mask': inputs.get('pos_attention_mask')}
            hard_neg = {'input_ids': inputs.get('hard_neg_input_ids'),
                       'attention_mask': inputs.get('hard_neg_attention_mask')}

            layer_losses = []
            for layer_idx in self.layers_to_use:
                with torch.no_grad():
                    query_embed = self.encode(model, query, layer_idx=layer_idx)

                if layer_idx != -1:
                   query_embed = query_embed.detach() # Detach query, except for the last layer
                
                pos_embed = self.encode(model, pos, layer_idx=layer_idx)
                hard_neg_embed = self.encode(model, hard_neg, layer_idx=layer_idx)

                layer_losses.append(self.info_nce(query_embed, pos_embed, hard_neg_embed))

            contrastive_loss = sum(layer_losses) / len(layer_losses)  # Average layer losses

        # Combine losses using the specified weights
        total_loss = (self.lm_weight * lm_loss) + (self.contrastive_weight * contrastive_loss)

        if return_outputs:
            return (total_loss, outputs)
        return total_loss