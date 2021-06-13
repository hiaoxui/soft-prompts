from typing import *

import torch
from torch import nn
from transformers import PreTrainedModel

from .. import util


class LanguageModel:
    def __init__(self, device, model: PreTrainedModel):
        self.device = device
        self.model = model
        self.model.to(device)

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def n_layer(self):
        raise NotImplementedError

    @property
    def emb(self):
        raise NotImplementedError

    def sample(self, **kwargs):
        raise NotImplementedError

    def fix(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

    def prepare_inputs(
            self,
            word_vector_list: List[torch.Tensor],
            label_mask_list: List[torch.Tensor],
            label_list: List[torch.Tensor],
            bias_list: List[torch.Tensor],
    ):
        ret = dict()
        bos_emb, eos_emb = self.emb(torch.tensor([util.tokenizer.encode(' ')], device=self.device))[0]
        pad_emb, = self.emb(torch.tensor([util.tokenizer.pad_token_id], device=self.device))
        max_len = max(map(len, label_list))
        pad_mask = torch.zeros([len(word_vector_list), max_len+2], dtype=torch.bool, device=self.device)
        for i, (word_vector, labels, label_mask, bias) in enumerate(
                zip(word_vector_list, label_list, label_mask_list, bias_list)
        ):
            pad_len = max_len - len(labels)
            if pad_len != 0:
                pad_mask[i, -pad_len:] = True
            word_vector_list[i] = torch.cat([
                bos_emb.unsqueeze(0), word_vector, torch.stack([eos_emb] + [pad_emb]*pad_len)
            ], dim=0)
            # Bias shape [#layer, #token, dim]
            bias_list[i] = torch.cat([
                torch.zeros([1, *bias.shape[1:]], device=self.device, dtype=torch.bool),
                bias,
                torch.zeros([pad_len+1, *bias.shape[1:]], device=self.device, dtype=torch.bool),
            ], dim=0)
            label_list[i] = torch.cat([
                torch.full([1], -100, device=self.device, dtype=torch.int64),
                labels,
                torch.full([pad_len+1], -100, device=self.device, dtype=torch.int64)
            ], dim=0)
            label_mask_list[i] = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.bool),
                label_mask,
                torch.zeros(pad_len+1, device=self.device, dtype=torch.bool),
            ], dim=0)
        ret['word_vectors'] = torch.stack(word_vector_list)
        ret['labels'] = torch.stack(label_list)
        ret['label_mask'] = torch.stack(label_mask_list)
        ret['attention_mask'] = (~pad_mask) * 1.0
        ret['bias'] = torch.stack(bias_list)
        assert len(set(map(lambda x: x.shape[:2], ret.values()))) == 1
        return ret

    def _auxiliary_tensors(self, label_mask):
        batch_size, max_token = label_mask.shape
        n_label = label_mask.sum(dim=1)
        max_label = int(n_label.max())
        target_mask = (torch.arange(0, max_label, device=self.device, dtype=torch.int).repeat(batch_size)
                       .reshape(batch_size, max_label).T < n_label).T
        label_indices = torch.full_like(target_mask, dtype=torch.int64, fill_value=max_token-1)
        # shape: [batch_size, max_label]
        label_indices[target_mask] = torch.arange(
            0, max_token, dtype=torch.int64, device=self.device
        ).repeat(batch_size).reshape(batch_size, max_token)[label_mask]
        return n_label, max_label, target_mask, label_indices

    def relax_forward(
            self,
            word_vectors: torch.Tensor,
            label_mask: Optional[torch.Tensor],
            weights: Optional[torch.Tensor],
            bias: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        :param word_vectors: A list of embedding seqs. Size [batch_size, sentence_length, emb_dim].
        :param label_mask: Same shape as word_emb. 0 is normal word and 1 is [MASK].
        Shape: [batch_size, max_length]
        :param weights: Shape: [batch_size]
        :param bias:
        :param labels: Correct token in the masked positions. If provided, a loss will be returned.
        Shape: [batch_size, max_token]
        :param attention_mask
        Shape: [batch_size, max_token]
        """
        batch_size, max_token, emb_dim = word_vectors.shape
        logits = self.model(
            inputs_embeds=word_vectors, attention_mask=attention_mask, bias=bias,
        )[0]
        word_dist = torch.softmax(logits, dim=2)

        loss = None
        log_target_dist = None
        if labels is not None and label_mask is not None and weights is not None:
            # Calculate weighted loss
            loss = 0.
            loss_fct = nn.CrossEntropyLoss()
            for logits_, label_mask_, weight_, labels_ in zip(logits, label_mask, weights, labels):
                local_loss = loss_fct(logits_[label_mask_], labels_[label_mask_])
                loss += weight_ * local_loss

        if labels is not None and label_mask is not None:
            # Calculate target probability, Shape: [batch_size, max_token]
            labels_clone = labels.clone()
            labels_clone[labels_clone == -100] = 0
            label_dist = word_dist.gather(dim=2, index=labels_clone.unsqueeze(2)).reshape(labels.shape)
            label_dist[~label_mask] = 1.
            log_target_dist = label_dist.log().sum(dim=1)

        return dict(
            loss=loss,
            word_dist=word_dist,
            num_tokens=int(label_mask.sum().cpu().detach()),
            log_target_dist=log_target_dist,
            label_mask=label_mask,
        )
