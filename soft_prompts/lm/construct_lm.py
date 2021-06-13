from .language_model import LanguageModel
from .bert_port import PreTrainedBert


def construct_lm(model_type, param_name, device, **kwargs) -> LanguageModel:
    if model_type in ['bert', 'roberta']:
        return PreTrainedBert(model_type, param_name, device, **kwargs)
    elif model_type in ['bart']:
        raise NotImplementedError
    else:
        raise NotImplementedError
