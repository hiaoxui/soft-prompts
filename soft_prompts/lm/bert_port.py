from transformers import (
    BertTokenizer,
    RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
)
from .modeling_bert import BertConfig, BertForMaskedLM
from torch import nn
import numpy as np

from ..util import *
from .. import util
from .language_model import LanguageModel


class PreTrainedBert(LanguageModel):
    """
    PreTrained language model.
    The parameters of this model should be fixed. It should only be used to compute the
    likelihood of a sentence.
    """
    def __init__(
            self,
            model_type: str = 'bert',
            param_name: str = 'bert-large-cased',
            device: str = 'cuda:0',
            max_length: int = 2048,
            batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_type: E.g., xlnet, bert
            param_name: E.g., xlnet-base-cased, bert-base-uncased
            device: cuda:0, cpu
            max_length: Maximum supported number of tokens. For memory concern.
            batch_size: Maximum sentences that could be processed in a batch.
        """
        model_classes = {
            'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
            'roberta': (RobertaConfig, BertForMaskedLM, RobertaTokenizer),
        }
        self.model_type = model_type
        self.config, model_class, tokenizer_class = model_classes[model_type]
        util.tokenizer = tokenizer_class.from_pretrained(param_name)
        model = model_class.from_pretrained(param_name)
        self.max_length = max_length
        self.batch_size = batch_size

        super().__init__(device, model)

    @property
    def n_layer(self):
        return self.model.config.num_hidden_layers

    @property
    def dim(self):
        return self.model.config.hidden_size

    @property
    def emb(self) -> nn.Embedding:
        emb = self.model.base_model.embeddings.word_embeddings
        return emb
