from typing import Optional, List, Dict
import json
from tqdm import tqdm

from ..components import *
from .. import util


class TREx(RelationDatabase):
    def __init__(
            self,
            files: Dict[str, str],
            relation_type_filter: Optional[List[str]] = None,
            single_lexicon: bool = False,
            vocab_file: str = None,
    ):
        vocab_list = None
        if vocab_file is not None:
            vocab_list = open(vocab_file).read().split('\n')
        super().__init__()
        self.single_lexicon = single_lexicon
        for predicate_name, fn in tqdm(list(files.items())):
            if relation_type_filter is not None and predicate_name not in relation_type_filter:
                continue
            rb = RelationBank(predicate_name, 2)
            self.banks[predicate_name] = rb
            pb = PatternBank(predicate_name)
            self.gold_pattern[predicate_name] = pb

            with open(fn, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                raw = json.loads(line)
                obj = raw['obj']
                sub = raw['sub']
                if self.single_lexicon and len(util.tokenizer.tokenize(obj)) > 1:
                    continue
                if vocab_list is None or obj in vocab_list:
                    rb.add_relation([sub, obj], 'TREx', raw.get('uuid'))
