from typing import *
import os

from ..components import Pattern, PatternBank
import json


def load_templates_single_file(
        path: str,
        relation_type_filter: Optional[List[str]] = None
):
    templates = list(map(json.loads, open(path).readlines()))
    all_banks = dict()
    for template in templates:
        text = template['template']
        pat = Pattern(text.split())
        if relation_type_filter is not None and template['relation'] not in relation_type_filter:
            continue
        pat.label_slot(['[X]'], 0)
        pat.label_slot(['[Y]'], 1)
        pb = PatternBank(template['relation'])
        pb.add(pat)
        all_banks[template['relation']] = pb
    return all_banks


def load_templates_folder(
        path: str,
        relation_type_filter: Optional[List[str]] = None
):
    all_banks = dict()
    for fn in os.listdir(path):
        rel_type = fn.split('.')[0]
        if relation_type_filter is not None and rel_type not in relation_type_filter:
            continue
        templates = list(map(json.loads, open(os.path.join(path, fn)).readlines()))
        pb = PatternBank(rel_type)
        for template in templates:
            text = template['template']
            pat = Pattern(text.split())
            pat.label_slot(['[X]'], 0)
            pat.label_slot(['[Y]'], 1)
            pb.add(pat)
        all_banks[rel_type] = pb
    return all_banks


def load_templates(
        path: str,
        relation_type_filter: Optional[List[str]] = None,
        min_length: int = -1,
        max_length: int = -1,
        max_num: int = -1,
):
    if os.path.isdir(path):
        db = load_templates_folder(path, relation_type_filter)
    else:
        db = load_templates_single_file(path, relation_type_filter)
    for rel_type, pb in db.items():
        pb.shrink(min_length, max_length, max_num)
    return db
