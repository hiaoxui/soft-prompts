import os

from .trex_reader import TREx


def load_db_general(path, **kwargs):
    subs = os.listdir(path)
    ret = {sp: [] for sp in ['train', 'dev', 'test']}
    print('loading from', path)
    for sp in ret:
        files = {sub: os.path.join(path, sub, f'{sp}.jsonl') for sub in subs}
        ret[sp] = TREx(files, **kwargs)
    return ret
