import yaml
import logging
import pprint
import os
from pprint import pprint
from io import StringIO
from argparse import ArgumentParser
import json

from ..io import load_db_general
from ..corpus import load_templates
from ..trainer import RPTrainer
from ..lm import construct_lm


def read_kwargs():
    parser = ArgumentParser()
    parser.add_argument('param', metavar='PARAM_FILE', type=str)
    parser.add_argument('--batch_size_no_grad', type=int)
    parser.add_argument('--batch_size_grad', type=int)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--accumulate_gradient', type=int)
    parser.add_argument('--max_layer', type=int)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--penalty', type=float, nargs=5)
    parser.add_argument('--randomize_prompt', type=bool)
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--conditional_prompt', type=bool)

    parser.add_argument('--model_type', type=str)
    parser.add_argument('--param_name', type=str)

    parser.add_argument('--template_path', type=str)
    parser.add_argument('--db_path', type=str)

    args = parser.parse_args()
    kwargs = yaml.load(open(args.param), Loader=yaml.FullLoader)

    for key in [
        'batch_size_no_grad', 'batch_size_grad', 'max_epoch', 'patience', 'accumulate_gradient', 'max_layer',
        'log_path', 'penalty', 'randomize_prompt', 'vocab_file', 'conditional_prompt'
    ]:
        val = getattr(args, key)
        if val is not None:
            print('override', key, val)
            kwargs['trainer'][key] = val

    for key in ['model_type', 'param_name']:
        val = getattr(args, key)
        if val is not None:
            print('override', key, val)
            kwargs['lm'][key] = val

    for key in ['db_path', 'template_path']:
        val = getattr(args, key)
        if val is not None:
            print('load from', val)
            kwargs[key.split('_')[0]]['path'] = val

    pprint(kwargs)
    return kwargs


def run():
    kwargs = read_kwargs()

    log_path = kwargs.get('trainer').get('log_path')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level='INFO',
        handlers=[
            logging.FileHandler(os.path.join(log_path, 'stdout.txt')),
            logging.StreamHandler(),
        ]
    )

    json.dump(kwargs, open(os.path.join(log_path, 'config.json'), 'w'))

    lm = construct_lm(**kwargs.pop('lm'))
    relation_db = load_db_general(**kwargs.get('db'))
    pattern_banks = load_templates(**kwargs.pop('template'))
    trainer = RPTrainer(lm, kwargs.pop('trainer'))

    string_out = StringIO()
    pprint(pattern_banks, string_out)
    logging.info(string_out.getvalue())

    trainer.experiment(pattern_banks, relation_db)


if __name__ == '__main__':
    run()
