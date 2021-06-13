from __future__ import annotations
from typing import *
from abc import ABC
import random
import uuid
import copy

import numpy as np

from ..components.pattern import PatternBank
from .. import util


class RelationInstance:
    """
    A relation instance is a relation type with a list of entities.
    """
    def __init__(self, relation_type: str, entities: List[str], source: str, uuid_: Optional[str] = None) -> NoReturn:
        """
        Args:
            relation_type: The specified relation type.
            entities: The number of entities should be equal to the valence of relation type.
            source: See RelationBank.
        """
        self.relation_type = relation_type
        self.entities = entities
        # TODO: How many source types do we need?
        self.source = source
        if uuid_ is None:
            self.uuid = uuid.uuid4()
        else:
            self.uuid = uuid.UUID(uuid_)

    def __repr__(self) -> str:
        """
        Human readable description.
        """
        return f'<{self.relation_type}: {str(self.entities)}>'


class RelationBank:
    """
    Relation Bank is a collection of relations (relation instances).
    """
    def __init__(self, relation_type: str, valence: int, relation_instances: Optional[List[RelationInstance]] = None):
        self.relation_type = relation_type
        self.valence = valence
        self.bank: List[RelationInstance] = relation_instances or list()
        self.gold_patterns: List[Pattern] = list()

    def add_relation(self, entities: List[str], source: str, uuid_: Optional[str] = None) -> NoReturn:
        """
        Add a relation.
        Args:
            entities: A list of entities, each of which should be a string.
            source: Indicate where this is from. TODO: Design a schema.
            uuid_: UUID.
        """
        assert self.valence == len(entities)
        new_instance = RelationInstance(self.relation_type, entities, source, uuid_)
        self.bank.append(new_instance)

    def __len__(self) -> int:
        """
        Returns:
            Number of relations.
        """
        return len(self.bank)

    def __repr__(self) -> str:
        """
        Human readable description.
        """
        return f'<Relation bank for {self.relation_type} with {len(self)} instances.>'

    def __iter__(self) -> Iterator[RelationInstance]:
        """
        Iterate all relations.
        """
        yield from self.bank

    def sample(self, n: int) -> List[RelationInstance]:
        """
        Sample n relations without replacement.
        If less than n examples are stored, will only return n samples.
        Args:
            n: Number of samples.

        Returns:
            Sampled relations.
        """
        indices = list(range(min(len(self), n)))
        random.shuffle(indices)
        samples = list()
        for idx in indices[:n]:
            samples.append(self.bank[idx])
        return samples

    def subset(self, start=0, end=None, max_size=None) -> RelationBank:
        ret = RelationBank(self.relation_type, self.valence)
        end = end or len(self.bank)
        if max_size is not None:
            end = min(start+max_size, end)
        ret.bank = self.bank[start:end]
        return ret

    def __getitem__(self, idx: int) -> RelationInstance:
        return self.bank[idx]

    def blank_set(self, idx: int):
        ret_set = set()
        for rel in self.bank:
            ret_set.add(rel.entities[idx])
        return ret_set

    def y_dist(self, plus_smoothing: float = 0.) -> np.ndarray:
        cnt = np.zeros(shape=[len(util.tokenizer.get_vocab())], dtype=np.int)
        for rel in self.bank:
            y_idx = util.tokenizer.encode(rel.entities[1], add_special_tokens=False)
            assert len(y_idx) == 1
            cnt[y_idx[0]] += 1
        cnt = cnt.astype(dtype=np.float)
        cnt += plus_smoothing
        dist = cnt / cnt.sum()
        return dist

    def clone(self) -> "RelationBank":
        ret = RelationBank(self.relation_type, self.valence)
        for ins in self.bank:
            ret.add_relation(copy.deepcopy(ins.entities), 'clone')
        return ret


class RelationDatabase(ABC):
    def __init__(self):
        self.banks: Dict[str, RelationBank] = dict()
        self.gold_pattern: Dict[str, PatternBank] = dict()
        # There might be some sentences in the database representing the relation.

    def __repr__(self) -> str:
        return f"Relation Database with {len(self.banks)} relation types and {self.n_instance} instances in total."

    @property
    def n_instance(self) -> int:
        return sum(map(len, self.banks.values()))

    def report(self, limit: int = -1) -> None:
        for idx, bank in enumerate(self.banks.values()):
            print(f'Relation bank {bank.relation_type} contains {len(bank)} instances.')
            if idx == limit:
                break

    def __getitem__(self, item: str):
        return self.banks[item]

    def filter(self, relation_list: List[str]):
        banks_ = dict()
        for rel in relation_list:
            if rel in self.banks:
                banks_[rel] = self.banks[rel]
        self.banks = banks_

    @property
    def relations(self) -> List[str]:
        return list(self.banks.keys())
