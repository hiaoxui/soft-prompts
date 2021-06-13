from __future__ import annotations
from typing import *
import logging
import random
from itertools import product
from collections import defaultdict

import torch
from torch import nn
import numpy as np

from ..components import *
from ..lm import LanguageModel
from ..util import batch
from .. import util


logger = logging.getLogger(__name__)


class PatternModel:
    def __init__(
            self,
            pattern_bank: PatternBank,
            device: str,
            lm: LanguageModel,
            max_layer: int,
            gen_emb: bool = True,
            force_single_token: bool = False,
            vocab_file: str = None,
            conditional_prompt: bool = False,
    ) -> None:
        self.device = device

        self.pattern_bank = pattern_bank.hollow()
        self.lm = lm
        self.lm.fix()

        if gen_emb:
            self.pattern_bank.gen_relaxed_emb(self.emb, min(lm.n_layer, max_layer), lm.dim, device)

        # Not for back-prop. Shape: [#pat]
        self.weights = torch.ones(size=[len(self)], requires_grad=False, device=self.device) / len(self)

        # Don't train word_embedding
        self.emb.weight.requires_grad_(False)

        # Length distribution
        self.blank_length_voc = list()
        self.blank_length_prob: Optional[np.ndarray] = None

        self.force_single_token = force_single_token
        self.conditional_prompt = conditional_prompt

        # Restrict vocab
        if vocab_file is not None:
            limited_vocab = open(vocab_file).read().split('\n')
            mask_ids = util.tokenizer.convert_tokens_to_ids(limited_vocab)
            assert len(limited_vocab) == len(mask_ids)
            self.vocab_mask = torch.zeros([util.tokenizer.vocab_size], device=self.device, dtype=torch.bool)
            self.vocab_mask[mask_ids] = True
        else:
            self.vocab_mask = torch.ones([util.tokenizer.vocab_size], device=self.device, dtype=torch.bool)

    def randomize_prompt(self):
        mean, std = self.emb.weight.mean(0), self.emb.weight.std(0)
        for pat in self.pattern_bank:
            for idx in range(len(pat.vector)):
                pat.vector[idx, :] = torch.normal(mean, std)
            pat.vector = pat.vector.detach().clone()
            pat.vector.requires_grad_(True)

    def __len__(self) -> int:
        return len(self.pattern_bank)

    @property
    def emb(self) -> nn.Embedding:
        return self.lm.emb

    @property
    def emb_dim(self) -> int:
        return self.emb.embedding_dim

    def parameters(self):
        for pat in self.pattern_bank:
            # yield pat.vector
            yield pat.bias

    def iter_pattern_relation(
        self,
        rel: Union[Iterable[RelationInstance], RelationInstance, List[List[str]]],
        batch_size: int,
        shuffle: bool,
        iter_method: str,
        responsibility: Optional[torch.Tensor] = None,
        tqdm_desc: Optional[str] = None,
        slot_mask: Optional[List[bool]] = None,
    ) -> Iterable[Dict[str, Union[torch.Tensor, int]]]:
        """
        :param rel:
        :param batch_size:
        :param shuffle:
        :param iter_method:
        :param slot_mask:
        :param responsibility: Shape [#rel, #pat]
        :param tqdm_desc:
        :return:
        """
        if isinstance(rel, RelationInstance):
            rel = [rel]
        if responsibility is not None:
            assert responsibility.shape == (len(rel), len(self.pattern_bank))

        iter_func = {'zip': zip, 'product': product}[iter_method]
        if iter_method == 'zip':
            assert len(rel) == len(self.pattern_bank)
        rel_indices, pat_indices = list(range(len(rel))), list(range(len(self.pattern_bank)))
        to_iter = list(iter_func(rel_indices, pat_indices))
        if shuffle:
            random.shuffle(to_iter)
        for rel_pat_indices in batch(to_iter, batch_size, use_tqdm=tqdm_desc is not None, tqdm_desc=tqdm_desc):
            weights, word_vector_list, label_list, label_mask_list, bias_list = list(), list(), list(), list(), list()
            for rel_idx, pat_idx in rel_pat_indices:
                ri = rel[rel_idx]
                if isinstance(ri, RelationInstance):
                    entities = ri.entities
                else:
                    entities = ri
                pat = self.pattern_bank.bank[pat_idx]
                word_vector, labels, label_mask, bias_vector = fill_relaxed_pattern(
                    pat, entities, pat.bias, self.emb, self.device, slot_mask
                )
                word_vector_list.append(word_vector)
                bias_list.append(bias_vector)
                label_list.append(labels)
                label_mask_list.append(label_mask)
                if responsibility is not None:
                    weights.append(float(responsibility[rel_idx, pat_idx]))
            inputs = self.lm.prepare_inputs(word_vector_list, label_mask_list, label_list, bias_list)
            if responsibility is not None:
                weight_tensor = torch.tensor(weights, device=self.device)
            else:
                weight_tensor = None
            ret = self.lm.relax_forward(
                weights=weight_tensor,
                **inputs
            )
            yield ret

    def compute_responsibility(
        self,
        relation_bank: RelationBank,
        batch_size: int,
    ) -> torch.Tensor:
        n_rel = len(relation_bank)
        ret_list = list()
        with torch.no_grad():
            for rst in self.iter_pattern_relation(
                relation_bank, batch_size, False, 'product', slot_mask=[False, True]
            ):
                ret_list.append(rst['log_target_dist'])
        ret = torch.cat(ret_list, dim=0)
        log_res = ret.reshape(n_rel, len(self.pattern_bank))
        log_res = log_res + self.weights.log()
        log_res = log_res - log_res.logsumexp(dim=1, keepdim=True)
        res = log_res.exp()
        # Shape: [#rel, #pat]
        return res

    def compute_precision(self, relation_bank: RelationBank, rank: int):
        entities = [x.entities for x in relation_bank.bank]
        predictions = []
        n_correct = 0
        for ret, (_, answer) in zip(
                self.iter_pattern_relation(entities, 1, False, 'product', None, slot_mask=[False, True]), entities
        ):
            topk = ret['word_dist'][0].topk(rank)[1].tolist()
            label_mask = ret['label_mask'][0]
            assert int(label_mask.sum()) == 1
            label_idx = int(torch.arange(len(label_mask), device=label_mask.device)[label_mask])
            pred = topk[label_idx]
            predictions.append(pred)
            gold = util.tokenizer._convert_token_to_id(answer)
            if gold in pred:
                n_correct += 1
        return n_correct / len(entities)

    def compute_pattern_ppl(self) -> Tuple[torch.Tensor, float]:
        relation_entities = [pat.rel_in_corpus for pat in self.pattern_bank]
        pat_ppl = list()
        total_log_ppl = list()
        total_token = 0
        with torch.no_grad():
            for ret in self.iter_pattern_relation(relation_entities, 1, False, 'zip', None, 'Pattern PPL'):
                log_ppl = ret['log_target_dist'][0]
                log_ppl = log_ppl / ret['num_tokens']
                total_token += ret['num_tokens']
                pat_ppl.append(float(torch.exp(-log_ppl)))
                total_log_ppl.append(float(ret['log_target_dist'][0]))
        pat_ppl = torch.tensor(pat_ppl, device=self.device)
        avg_ppl = float(np.exp(-np.sum(total_log_ppl) / total_token))
        return pat_ppl, avg_ppl

    def fit_weight(
        self,
        responsibility: torch.Tensor
    ) -> NoReturn:
        weights = responsibility.sum(dim=0)
        self.weights = weights / weights.sum()

    def effective_pattern_num(self):
        # return float(self.weights.sum()**2 / (self.weights**2).sum())
        return float(torch.exp(-torch.log(self.weights) @ self.weights))

    def dump(self):
        return {
            'weights': self.weights.cpu().detach(),
            'pattern_bank': self.pattern_bank.dump()
        }

    def load(self, states):
        self.weights: torch.Tensor = states['weights'].to(device=self.device)
        self.pattern_bank = states['pattern_bank']
        self.pattern_bank.to_device(self.device)

    def collect_blank_length(self, relation_bank: RelationBank):
        cnt = defaultdict(int)
        for ri in relation_bank:
            lengths = tuple(len(util.tokenizer.encode(ent))-2 for ent in ri.entities)
            cnt[lengths] += 1
        cnt = list(cnt.items())
        cnt.sort(key=lambda x: x[1])
        prob = list()
        for lengths, c in cnt:
            self.blank_length_voc.append(lengths)
            prob.append(c / len(relation_bank))
        self.blank_length_prob = np.array(prob)

    def sample_lengths(self):
        sample_vec = np.random.multinomial(1, self.blank_length_prob)
        select_idx = int(np.arange(len(self.blank_length_voc))[sample_vec.astype(np.bool)])
        select = self.blank_length_voc[select_idx]
        if self.force_single_token:
            select = [1 for _ in select]
        return select

    def top_by_weights(self, weights: torch.Tensor, top: int, verbose: int, additional_info=""):
        top_k = torch.argsort(-weights)[:top].cpu().tolist()
        if verbose > 0:
            logger.info(f'Top {top} pattern list ({additional_info}): {top_k}')
        if verbose > 1:
            logger.info('They are:')
            for idx in top_k:
                pat = self.pattern_bank[idx]
                to_print = f'No. {idx}: {pat}.'
                if verbose > 2:
                    to_print += f'Original entities: {pat.rel_in_corpus}'
                logger.info(to_print)
        return top_k

    def conditional_generate_single_slot(
            self,
            batch_size: int,
            rel_bank: RelationBank,
            freq: Optional[np.ndarray] = None,
    ):
        single_option = self.force_single_token
        answers = torch.tensor(util.tokenizer.convert_tokens_to_ids([rel_ins.entities[1] for rel_ins in rel_bank]))

        self.force_single_token = True
        all_dist = list()
        with torch.no_grad():
            for ret in self.iter_pattern_relation(rel_bank, batch_size, False, 'product', None, None, [False, True]):
                lm = ret['label_mask']
                target_mask = torch.arange(lm.shape[1], device=lm.device).unsqueeze(0).expand_as(lm)[lm]
                target_dist = ret['word_dist'].gather(
                    1, target_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, ret['word_dist'].shape[2])
                ).squeeze(1)
                all_dist.append(target_dist.detach().clone().cpu())
        all_dist = torch.cat(all_dist, 0)
        all_dist.T[~self.vocab_mask] = 0.0
        all_dist = all_dist.reshape([len(answers), -1, all_dist.shape[1]]).log()
        weights = self.weights
        if self.conditional_prompt:
            weights = weights * self.conditional_prompt_prob(batch_size, rel_bank)
        all_dist = (all_dist.permute(0, 2, 1) + weights.cpu().log()).permute(0, 2, 1)
        target_dist = all_dist.logsumexp(1)
        if freq is not None:
            freq_dist = torch.tensor(freq, dtype=torch.float32)
            target_dist = freq_dist.unsqueeze(0).log() + target_dist
        pred = (-target_dist).argsort(1)
        answer_ranks = (pred.T == answers).T
        answer_ranks = torch.arange(answer_ranks.shape[1]).unsqueeze(0).expand_as(answer_ranks)[answer_ranks]+1
        topk = (
                torch.arange(pred.shape[1]).unsqueeze(0).expand(answer_ranks.shape[0], -1) >=
                answer_ranks.unsqueeze(1).expand(-1, pred.shape[1])
        ).sum(0)
        self.force_single_token = single_option
        # The following line is super time consuming!
        # pred = [util.tokenizer.convert_ids_to_tokens(line) for line in pred]
        return pred, answer_ranks, topk

    def sample_entities(
            self,
            batch_size: int,
            output_format: str = 'JSON',  # Or tsv
    ):
        word_vector_list, label_mask_list, label_list, length_list, token_id_list = \
            list(), list(), list(), list(), list()
        selected_pattern_idx = list()
        for i in range(batch_size):
            pat_idx = int(self.weights.multinomial(1)[0])
            selected_pattern_idx.append(pat_idx)
            lengths = self.sample_lengths()
            length_list.append(lengths)
            pat = self.pattern_bank[pat_idx]
            entities = [' '.join([util.tokenizer.mask_token]*le) for le in lengths]
            word_vector, labels, label_mask = fill_relaxed_pattern(pat, entities, self.emb, self.device)
            word_vector_list.append(word_vector)
            label_mask_list.append(label_mask)
            label_list.append(labels)
            token_id_list.append(pat.token_ids)
        inputs = self.lm.prepare_inputs(word_vector_list, label_mask_list, label_list, token_id_list)
        with torch.no_grad():
            samples = self.lm.sample(**inputs)
        rst = list()

        for pat_idx, lengths, sample in zip(selected_pattern_idx, length_list, samples):
            cur = list()
            i = 0
            for length_ent in lengths:
                entity = util.tokenizer.convert_tokens_to_string(sample[i:length_ent+i]).strip()
                cur.append(entity)
                i += length_ent
            cur.append(str(self.pattern_bank[pat_idx]))
            rst.append(cur)

        if output_format == 'JSON':
            return rst

        return '\n'.join(['\t'.join(line) for line in rst])

    def conditional_prompt_prob(
            self,
            batch_size: int,
            rel_bank: RelationBank,
    ):
        masked_rel_bank = rel_bank.clone()
        for rel_ins in masked_rel_bank:
            rel_ins.entities[1] = util.tokenizer.mask_token
        all_log_dist = []
        with torch.no_grad():
            for ret in self.iter_pattern_relation(masked_rel_bank, batch_size, False, 'product', None, None, [True, False]):
                all_log_dist.append(ret['log_target_dist'])
        log_dist = torch.cat(all_log_dist)
        log_dist: torch.Tensor = log_dist.reshape([len(rel_bank), -1])
        log_dist = log_dist.sum(dim=0)
        prob = torch.softmax(log_dist, 0)
        return prob
