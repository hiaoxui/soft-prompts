from __future__ import annotations
from typing import *
from collections import defaultdict
from itertools import takewhile, accumulate
from operator import itemgetter
import logging
import random
import uuid
import os
import time
import pickle

import torch
from torch import optim
from torch.optim.optimizer import Optimizer
import numpy as np

from ..model import PatternModel
from ..components import RelationBank, RelationDatabase, PatternBank
from ..lm import LanguageModel


logger = logging.getLogger('Trainer')


class RPTrainer:
    def __init__(
        self,
        lm: LanguageModel,
        trainer: dict,
        **kwargs
    ):
        random_seed = trainer.pop('random_seed', 1)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.device = trainer.pop('device')

        self.lm = lm

        self.batch_size_no_grad = trainer.pop('batch_size_no_grad')
        self.batch_size_grad = trainer.pop('batch_size_grad')
        self.patience = trainer.pop('patience')
        self.max_epoch = trainer.pop('max_epoch')

        self.train_ratio = trainer.pop('train_ratio', 0.8)
        self.dev_ratio = trainer.pop('dev_ratio', 0.1)
        self.max_train = trainer.pop('max_train', None)
        self.max_dev = trainer.pop('max_dev', None)
        self.max_test = trainer.pop('max_test', None)
        self.shuffle = trainer.pop('shuffle')
        self.weight_only_epoch = trainer.pop('weight_only_epoch')
        self.force_single_token = trainer.pop('force_single_token', False)
        self.show_top_patterns = trainer.pop('show_top_patterns', 0)
        self.distinct_slot_idx = trainer.pop('distinct_slot_idx', None)
        self.fix_weights = trainer.pop('fix_weights', False)
        self.randomize_prompt = trainer.pop('randomize_prompt', False)
        self.log_path = trainer.pop('log_path')
        os.makedirs(self.log_path, exist_ok=True)

        self.leave_out = trainer.pop('leave_out')
        self.training_target = trainer.pop('training_target')
        self.accumulate_gradient = trainer.pop('accumulate_gradient')

        self.max_layer = trainer.pop('max_layer')
        self.penalty = trainer.pop('penalty')
        if self.penalty is not None:
            self.penalty = torch.tensor(self.penalty, device=self.device)

        self.frequent = trainer.pop('frequent', False)
        self.smoothing_factor = trainer.pop('smoothing', False)
        self.vocab_file = trainer.pop('vocab_file', None)
        self.conditional_prompt = trainer.pop('conditional_prompt', False)

        for key in kwargs:
            logger.warning(f'{key} in config not used.')

    def em_step(
        self,
        model: PatternModel,
        relation_bank: RelationBank,
        responsibility: Optional[torch.Tensor],
        opt: Optimizer,
        associated_relation_uuid: List[uuid.UUID],
        train_emb: bool,
    ):
        """

        :param model:
        :param relation_bank:
        :param responsibility:
        :param opt:
        :param associated_relation_uuid:
        length = #pat
        :param train_emb:
        :return:
        """
        # E step
        if responsibility is None:
            responsibility = model.compute_responsibility(relation_bank, self.batch_size_no_grad)

        # M step
        # Latent variable update
        if not self.fix_weights:
            model.fit_weight(responsibility)
            logger.info('Weight training done.')

        # Gradient Ascent
        if train_emb:
            # Shape [#rel, #pat]
            weights = responsibility.clone()
            if self.leave_out:
                for i, rel in enumerate(relation_bank):
                    ignore_mask = [u_ == rel.uuid for u_ in associated_relation_uuid]
                    ignore_mask = torch.tensor(ignore_mask, device=self.device)
                    weights[i, ignore_mask] = 0.

            slot_masks = list()
            if 'conditional' in self.training_target:
                slot_masks.append([False, True])
            if 'generate' in self.training_target:
                slot_masks.append([True, True])

            gradient_count = 0
            for slot_mask in slot_masks:
                for ret in model.iter_pattern_relation(
                    relation_bank, self.batch_size_grad, self.shuffle, 'product', weights, tqdm_desc='Gradient Ascent',
                    slot_mask=slot_mask
                ):
                    gradient_count += 1
                    loss = ret['loss']
                    loss = loss + model.pattern_bank.regularization() @ self.penalty
                    loss.backward()
                    if gradient_count % self.accumulate_gradient == 0:
                        opt.step()
                        opt.zero_grad()
            opt.step()
            opt.zero_grad()
            logger.info('Embedding training done.')

        # E step
        responsibility = model.compute_responsibility(relation_bank, self.batch_size_no_grad)
        # ppl = model.compute_ppl(relation_bank, self.batch_size_grad, responsibility)

        effective_pattern_num = model.effective_pattern_num()

        return responsibility, None, effective_pattern_num

    @staticmethod
    def overlapping_warning(
            train_set: RelationBank,
            dev_set: RelationBank,
            test_set: RelationBank,
            ent_idx: List[int]
    ):
        train_dev = train_test = 0
        train_ins = {tuple(ins.entities[ei] for ei in ent_idx) for ins in train_set}
        for dev_point in dev_set:
            if tuple(dev_point.entities[ei] for ei in ent_idx) in train_ins:
                train_dev += 1
        for test_point in test_set:
            if tuple(test_point.entities[ei] for ei in ent_idx) in train_ins:
                train_test += 1
        if train_dev != 0:
            logger.warning(f'Overlapping {tuple(ent_idx)} between train and dev: {train_dev} / {len(dev_set)}')
        if train_test != 0:
            logger.warning(f'Overlapping {tuple(ent_idx)} between train and test: {train_test} / {len(test_set)}')

    def split_relation_bank(self, relation_bank: RelationBank):
        random.shuffle(relation_bank.bank)
        if self.distinct_slot_idx is None:
            train_size = int(len(relation_bank) * self.train_ratio)
            dev_size = int(len(relation_bank) * self.dev_ratio)
            train_set = relation_bank.subset(0, train_size, self.max_train)
            dev_set = relation_bank.subset(train_size, train_size+dev_size, self.max_dev)
            test_set = relation_bank.subset(train_size+dev_size, len(relation_bank), self.max_test)
        else:
            rel_groups = defaultdict(list)
            for rel_ins in relation_bank:
                rel_groups[rel_ins.entities[self.distinct_slot_idx]].append(rel_ins)
            rel_groups = list(rel_groups.items())
            if len(rel_groups) < 3:
                logger.warning('Too few groups to split.')
            train_size = len(list(takewhile(
                lambda x: x[0] < len(rel_groups)-2 and x[1] < int(len(relation_bank)*self.train_ratio),
                enumerate(accumulate(map(lambda x: len(x[1]), rel_groups)))
            )))
            dev_size = len(list(takewhile(
                lambda x: x[0] < len(rel_groups)-1 and x[1] < int(len(relation_bank)*(self.train_ratio+self.dev_ratio)),
                enumerate(accumulate(map(lambda x: len(x[1]), rel_groups)))
            ))) - train_size
            train_set = RelationBank(
                relation_bank.relation_type, relation_bank.valence,
                sum(map(itemgetter(1), rel_groups[:train_size]), [])
            ).subset(max_size=self.max_train)
            dev_set = RelationBank(
                relation_bank.relation_type, relation_bank.valence,
                sum(map(itemgetter(1), rel_groups[train_size:train_size+dev_size]), [])
            ).subset(max_size=self.max_dev)
            test_set = RelationBank(
                relation_bank.relation_type, relation_bank.valence,
                sum(map(itemgetter(1), rel_groups[train_size+dev_size:]), [])
            ).subset(max_size=self.max_test)
        self.overlapping_warning(train_set, dev_set, test_set, [0, 1])
        self.overlapping_warning(train_set, dev_set, test_set, [1])
        logger.info(f'Train/dev/test size: {len(train_set)}, {len(dev_set)}, {len(test_set)}')

        return train_set, dev_set, test_set

    def train_one_relation(
        self,
        pattern_bank: PatternBank,
        relation_bank_splits: List[RelationBank],
    ):
        responsibility = None
        patience = self.patience

        train_set, dev_set, test_set = relation_bank_splits
        logger.info(f'Training relation {relation_bank_splits[0].relation_type} with {len(pattern_bank)} patterns.')
        model = PatternModel(
            pattern_bank, self.device, self.lm, self.max_layer, force_single_token=self.force_single_token,
            vocab_file=self.vocab_file, conditional_prompt=self.conditional_prompt
        )
        model.collect_blank_length(train_set)
        if self.randomize_prompt:
            model.randomize_prompt()

        opt = optim.Adam(model.parameters())

        before_pred, answer_ranks, answer_topk = model.conditional_generate_single_slot(self.batch_size_no_grad, test_set)
        ret = {
            'before': {
                'top1': int(answer_topk[1]),
                'top10': int(answer_topk[10]),
                'answer_ranks': answer_ranks
            }
        }
        logger.info(f'Test precision before training: {int(answer_topk[1]) / len(test_set)}')
        best_precision, best_state = 0., model.dump()

        effective_pattern_num = 0
        logger.info('Start Training.')

        for i_epoch in range(self.max_epoch):
            logger.info(f'EM -- {i_epoch+1}-th epoch.')

            responsibility, train_ppl, effective_pattern_num = self.em_step(
                model, train_set, responsibility, opt, pattern_bank.relation_uuid_list,
                i_epoch >= self.weight_only_epoch
            )

            # dev_ppl = model.compute_ppl(dev_set, self.batch_size_no_grad)
            # logger.info(f'Dev PPL = {dev_ppl:.5f}')

            # logger.info(f'Train Perplexity: {train_ppl:.5f}')
            logger.info(f'Effective num of patterns: {effective_pattern_num}')

            dev_p1 = int(model.conditional_generate_single_slot(self.batch_size_no_grad, dev_set)[2][1]) / len(dev_set)
            logger.info(f'Dev precision = {dev_p1}')

            if dev_p1 > best_precision:
                logger.info('Best this epoch. Save.')
                best_precision = dev_p1
                patience = self.patience
                best_state = model.dump()
            else:
                if i_epoch >= self.weight_only_epoch and patience == 0:
                    logger.info('Early stop.')
                    break
                patience -= 1

            # if self.show_top_patterns > 0:
            #     model.top_by_weights(dev_ppl, self.show_top_patterns, 3)

        logger.info('Saving the best model to disk...')
        with open(os.path.join(self.log_path, f'model.{train_set.relation_type}.pkl'), 'wb') as fp:
            pickle.dump(best_state, fp)
        model.load(best_state)
        freq = train_set.y_dist(self.smoothing_factor) if self.frequent else None
        after_pred, answer_ranks, answer_topk = model.conditional_generate_single_slot(
            self.batch_size_no_grad, test_set, freq
        )
        ret['after'] = {
            'top1': answer_topk[1],
            'top10': answer_topk[10],
            'answer_ranks': answer_ranks,
        }
        ret['effective_pattern_num'] = effective_pattern_num
        logger.info(f'Test precision after training: {int(answer_topk[1]) / len(test_set)}')
        # ret['model'] = model

        return ret

    def experiment(
            self,
            pattern_db: Dict[str, PatternBank],
            relation_db: Union[Dict[str, RelationDatabase], RelationDatabase],
    ):
        returns = dict()
        n_cur, n_total = 0, len(pattern_db)
        for rel_type, pb in pattern_db.items():
            since = time.time()
            n_cur += 1
            logger.info(f'Now processing the {n_cur}-th / {n_total} relation.')
            if isinstance(relation_db, RelationDatabase):
                if rel_type not in relation_db.banks:
                    continue
                rb = relation_db[rel_type]
                splits = self.split_relation_bank(rb)
            else:
                splits = list()
                if rel_type not in relation_db['train'].banks:
                    continue
                for split in ['train', 'dev', 'test']:
                    splits.append(relation_db[split].banks[rel_type])
            if not all([len(one_db) > 0 for one_db in splits]):
                logger.warning(f'Skipping {rel_type} because it\' empty')
                continue
            returns[rel_type] = self.train_one_relation(pb, splits)
            logger.info(f'Time cost for {rel_type}: {time.time() - since:.2f}')

        pred_path = os.path.join(self.log_path, 'prediction.txt')
        if os.path.exists(pred_path): os.remove(pred_path)

        for state in ['before', 'after']:
            logger.info(f'Showing results {state} training:')
            answer_ranks = list()
            for rel_type, ret in returns.items():
                ar = ret[state]['answer_ranks']
                with open(pred_path, 'a') as fp:
                    fp.write('\t'.join([state, rel_type] + list(map(str, ar.tolist()))) + '\n')
                answer_ranks.append(ar)
                if state == 'after':
                    with open(pred_path, 'a') as fp:
                        fp.write(f"{rel_type}\t{ret['effective_pattern_num']}\n")

            answer_ranks = torch.cat(answer_ranks, 0)
            logger.info(f'P@1: {float((answer_ranks <= 1).sum()) / answer_ranks.shape[0]}')
            logger.info(f'P@10: {float((answer_ranks <= 10).sum()) / answer_ranks.shape[0]}')
            logger.info(f'P@100: {float((answer_ranks <= 100).sum()) / answer_ranks.shape[0]}')
            logger.info(f'MRR: {sum(1/answer_ranks.to(dtype=float)) / answer_ranks.shape[0]}')
