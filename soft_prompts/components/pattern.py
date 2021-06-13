from __future__ import annotations
from typing import *
from operator import itemgetter
import uuid

import torch
from torch import nn
from ..util import word2vec
from .. import util


class Pattern:
    """
    A pattern is a sentence with blanks.
    Although they're blanks, the original tokens will be preserved.
    I just use indices to indicate where the slots are.
    """
    def __init__(
        self,
        text: Union[list, str],
        n_slots: int = 2,
        associated_relation_uuid: Optional[str] = None
    ):
        """
        A pattern should start with a text, or a list of tokens.
        One and only one of them should be provided.
        Args:
            text: A string of sentence or a list of tokens.
            I will tokenize the string into tokens.
            n_slots: Number of slots. By default is 2, namely x and y.
        """
        # Separation of tokens.
        self.sep = ' '
        # Origin is the original text that this pattern comes from.
        # This shouldn't be changed even if the content in slots are replaced.
        if isinstance(text, str):
            self.tokens = util.tokenizer.tokenize(text)
            self.origin = text
        elif isinstance(text, list):
            self.tokens = text
            self.origin = self.sep.join(text)
        self.tokens = list(filter(lambda token: len(token) > 0, self.tokens))
        self.n_slots = n_slots
        # Start and end index for each slot.
        self.indices: List[Tuple[int, int]] = [(-1, -1) for _ in range(n_slots)]
        # The uuid of associated relation instance. By default it's None.
        if associated_relation_uuid is None:
            self.associated_relation_uuid: Optional[uuid.UUID] = uuid.uuid4()
        else:
            self.associated_relation_uuid: Optional[uuid.UUID] = None

        # Associated entity pairs in the corpus
        self.rel_in_corpus: Optional[List[str]] = None
        # Shape: [length, emb_dim]
        self.vector: Optional[torch.Tensor] = None
        self.ref_vector: Optional[torch.Tensor] = None
        self.is_slot: List[bool] = [False] * n_slots

        self.slot_vector = None
        self.bias = None
        self.token_ids = None

    def label_slot(self, phrase: Union[str, List[str]], target: int) -> bool:
        """
        Label a span as slot. I will check whether the phrase and the tokens match.
        Args:
            phrase: The phrase to be matched.
            target: The index of target slot.

        Returns:
            True if matched and no overlapping detected.

        """
        if isinstance(phrase, str):
            phrase_tokens = util.tokenizer.tokenize(phrase)
        else:
            phrase_tokens = phrase
        phrase_tab = '\t'.join(phrase_tokens)
        text_tab = '\t'.join(self.tokens)
        idx = text_tab.find(phrase_tab)
        if idx == -1:
            return False

        start = text_tab[:idx].count('\t')
        end = start + len(phrase_tokens)

        # Overlapping check
        indices = self.indices.copy()
        indices[target] = (start, end)
        indices.sort(key=lambda x: x[0])
        for t_idx in range(0, self.n_slots-1):
            if indices[t_idx][1] > indices[t_idx+1][0]:
                return False

        # commit
        self.indices[target] = (start, end)
        self.is_slot[target] = True
        return True

    def fill(
        self,
        new_phrase: Union[str, List[str]],
        target: int,
        keep_slot: bool = True,
        share_vector: bool = True,
    ) -> Pattern:
        """
        Fill a slot with a new phrase. Will create a new pattern instead of modifying
        current one.
        Args:
            new_phrase: The phrase to fill. It will be tokenized.
            target: The index of slot.
            keep_slot: Keep this as a slot even after filling.
            share_vector: Return a new word vector or share with the original one.

        Returns:
            A new pattern.

        """
        # Replace tokens with new phrase.
        start, end = self.indices[target]
        tokens = self.tokens.copy()
        if isinstance(new_phrase, list):
            new_tokens = new_phrase
        else:
            new_tokens = util.tokenizer.tokenize(new_phrase, remove_head=False)
        tokens = tokens[:start] + new_tokens + tokens[end:]

        # Create a new pattern for filling.
        pat = Pattern(tokens, self.n_slots)
        pat.origin = self.origin
        pat.indices = self.indices.copy()
        pat.rel_in_corpus = self.rel_in_corpus
        pat.associated_relation_uuid = self.associated_relation_uuid
        if share_vector:
            pat.vector = self.vector
        else:
            pat.vector = self.vector.clone()
        pat.ref_vector = self.ref_vector
        pat.is_slot = self.is_slot.copy()
        if not keep_slot:
            pat.is_slot[target] = False

        # The change of token length.
        len_delta = len(new_tokens) - (end-start)
        if len_delta != 0:
            # Update all slots after the target slot.
            pat.indices[target] = (self.indices[target][0], self.indices[target][1]+len_delta)
            for t_idx, (one_start, one_end) in enumerate(pat.indices):
                if one_start > end:
                    pat.indices[t_idx] = (one_start+len_delta, one_end+len_delta)

        return pat

    def phrase(self, target: int) -> str:
        """
        The surface form of the content of a slot.
        Output from the tokens, instead of original text.
        Args:
            target: The index of the slot.

        Returns:
            The content of this slot.

        """
        return self.sep.join(self.tokens[self.indices[target][0]: self.indices[target][1]])

    @property
    def text(self) -> str:
        """
        Returns:
            Concatenation of tokens.
        """
        return self.sep.join(self.tokens)

    def reset(self) -> None:
        """
        Reset all slots to default values (undecided)
        Returns:
            Nothing.
        """
        self.indices = [(-1, -1) for _ in range(self.n_slots)]

    def __repr__(self) -> str:
        """
        Human readable pattern.
        """
        indices = list(enumerate(self.indices.copy()))
        indices.sort(key=lambda pair: pair[1][0], reverse=True)
        tokens = self.tokens.copy()
        for t_idx, (start, end) in indices:
            for idx in range(end-start):
                del tokens[start]
            tokens.insert(start, f'[SLOT{t_idx}]')
        return util.tokenizer.convert_tokens_to_string(tokens)\
            .replace('[', ' [').replace(']', '] ').replace('  ', ' ').strip()

    def hollow(self) -> Pattern:
        """
        Fill the blanks with a placeholder token.
        """
        new_pat = self
        for i in range(self.n_slots):
            new_pat = new_pat.fill(["@@BLANK@@"], target=i)
        if self.rel_in_corpus is not None:
            new_pat.rel_in_corpus = self.rel_in_corpus.copy()
        return new_pat

    def __len__(self):
        return len(self.tokens)

    def dump(self):
        new_pat = Pattern(self.tokens, self.n_slots, self.associated_relation_uuid)
        new_pat.origin = self.origin
        new_pat.rel_in_corpus = self.rel_in_corpus
        new_pat.indices = self.indices
        if self.vector is not None:
            new_pat.vector = self.vector.detach().cpu().clone()
        if self.bias is not None:
            new_pat.bias = self.bias.detach().cpu().clone()
        return new_pat

    def bias_2(self):
        # bias shape: [#layer+1, #token, emb]
        ret = [self.bias[0].square().sum()]
        for layer_idx in range(0, 24, 6):
            if self.bias.shape[0] > layer_idx + 6:
                ret.append(self.bias[layer_idx+1:layer_idx+7].square().sum())
            else:
                ret.append(torch.tensor(0.0, device=self.bias.device))
        return torch.stack(ret)


class PatternBank:
    """
    Pattern Bank is collection of patterns.
    A pattern bank should serve only one relation type.
    """
    def __init__(self, relation: str) -> NoReturn:
        self.relation = relation
        self.bank: List[Pattern] = list()

    def add(self, pattern: Pattern) -> None:
        """
        Add a new pattern.
        Args:
            pattern: The pattern to add.
        """
        self.bank.append(pattern)

    def __getitem__(self, item: int):
        return self.bank[item]

    def __len__(self) -> int:
        """
        Returns:
            The number of patterns.
        """
        return len(self.bank)

    def __add__(self, other: PatternBank) -> PatternBank:
        assert other.relation == self.relation
        pb = PatternBank(self.relation)
        pb.bank.extend(self.bank)
        pb.bank.extend(other.bank)
        return pb

    def __iter__(self) -> Iterator[Pattern]:
        yield from self.bank

    def shrink(self, min_length: int = -1, max_length: int = 1024, max_num: Optional[int] = None) -> None:
        if max_length < 0:
            max_length = 1024
        self.bank = list(filter(lambda p: min_length <= len(p.tokens) <= max_length, self.bank))
        if max_num is not None and max_num > 0:
            self.bank = self.bank[:max_num]

    def __repr__(self) -> str:
        return f'Pattern bank for relation {self.relation} with {len(self)} examples.'

    def hollow(self) -> PatternBank:
        new_pb = PatternBank(self.relation)
        for pat in self:
            new_pb.add(pat.hollow())
        return new_pb

    def gen_relaxed_emb(
            self,
            emb: nn.Embedding,
            n_layer: int,
            dim: int,
            device: str,
    ) -> None:
        """
        Hollow before calling this function!
        """
        for pat in self.bank:
            pat_token_ids = util.tokenizer.convert_tokens_to_ids(pat.tokens)
            pat_emb = emb(torch.tensor(pat_token_ids, device=device)).clone().detach()
            pat.ref_vector = pat_emb.detach().clone()
            pat_emb.requires_grad_(False)
            pat.vector = pat_emb
            pat.bias = torch.zeros([n_layer+1, len(pat), dim], device=device, requires_grad=True)
            pat.token_ids = pat_token_ids

    @property
    def max_pat_len(self):
        return max(map(len, self.bank))

    @property
    def relation_uuid_list(self) -> List[Optional[uuid.UUID]]:
        return [pat.associated_relation_uuid for pat in self.bank]

    def dump(self):
        detached_bank = PatternBank(self.relation)
        for pat in self.bank:
            detached_bank.bank.append(pat.dump())
        return detached_bank

    def to_device(self, device: str):
        for pat in self.bank:
            if pat.vector is not None:
                pat.vector = pat.vector.to(device=device)
            if pat.bias is not None:
                pat.bias = pat.bias.to(device=device)

    def regularization(self):
        summands = []
        for pat in self.bank:
            summands.append(pat.bias_2())
        ret = torch.stack(summands).sum(0)
        # Return shape: [5] (0 for embedding, 1 for layer 1-6, 2 for layer 7-13, and so on)
        return ret


def fill_relaxed_pattern(
        pattern: Pattern,
        entity: List[str],
        bias: torch.Tensor,
        emb: nn.Embedding,
        device: str,
        slot_mask: Optional[List[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    No BOS or EOS added.
    :param pattern:
    :param entity:
    :param bias:
    :param emb:
    :param device:
    :param slot_mask:
    :return:
    """

    max_entity_length = 16  # AD-HOC

    mask_emb, = emb(torch.tensor([util.tokenizer.mask_token_id], device=device))
    insert_indices = list(map(itemgetter(0), pattern.indices))
    if slot_mask is None:
        slot_mask = [True for _ in entity]

    entity_tokens: List[List[int]] = [
        util.tokenizer.encode(ent, add_special_tokens=False)[:max_entity_length] for ent in entity
    ]
    entity_vectors = [word2vec(ts, device, emb).unbind(0) for ts in entity]
    labels = [-100] * (len(pattern))
    word_vector = list(pattern.vector)
    bias_vector_list = list(bias.unbind(1))
    for i, (is_slot, insert_idx) in sorted(enumerate(zip(slot_mask, insert_indices)), key=lambda x: -x[1][1]):
        # Del the place holder
        del labels[insert_idx]
        del word_vector[insert_idx]
        del bias_vector_list[insert_idx]
        # From rear to front
        for token_id, token_vec in zip(entity_tokens[i][::-1], entity_vectors[i][::-1]):
            bias_vector_list.insert(insert_idx, torch.zeros_like(bias_vector_list[0]))
            if is_slot:
                # Insert some [MASK]
                word_vector.insert(insert_idx, mask_emb)
                labels.insert(insert_idx, token_id)
            else:
                word_vector.insert(insert_idx, token_vec)
                labels.insert(insert_idx, -100)
    labels = torch.tensor(labels, device=device)
    label_mask = labels != -100
    word_vector = torch.stack(word_vector)

    return word_vector, labels, label_mask, torch.stack(bias_vector_list)
