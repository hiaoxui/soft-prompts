from typing import *

import torch
from transformers import PreTrainedTokenizer

from .batcher import batch

tokenizer: Optional[PreTrainedTokenizer] = None


def word2vec(
        tokens: Union[List[str], str],
        device: str,
        embedder: torch.nn.Embedding,
):
    if isinstance(tokens, str):
        tokens = tokenizer.tokenize(tokens)
    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    word_ids_tensor = torch.tensor(word_ids, device=device)
    pat_emb = embedder(word_ids_tensor).detach()
    return pat_emb
