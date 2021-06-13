from typing import Iterable, Iterator, List, TypeVar
from tqdm import tqdm


Item = TypeVar('Item')


def batch(
        objs: Iterable[Item],
        batch_size: int,
        rep: int = 1,
        use_tqdm: bool = False,
        tqdm_desc: str = '',
) -> Iterator[List[Item]]:
    """
    Args:
        objs: Objects to be batched.
        batch_size: Batch size.
        rep: Number of epochs.
        use_tqdm: True to turn on tqdm.
        tqdm_desc: Description.

    Returns:
        An iterator for each batch.
    """
    cur_batch = list()
    if use_tqdm:
        objs = tqdm(objs, desc=tqdm_desc)
    for _ in range(rep):
        for obj in objs:
            cur_batch.append(obj)
            if len(cur_batch) == batch_size:
                yield cur_batch
                cur_batch = list()
    if len(cur_batch) > 0:
        yield cur_batch
