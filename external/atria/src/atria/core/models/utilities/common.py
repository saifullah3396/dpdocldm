from typing import List

from atria.core.utilities.typing import BatchDict


def _validate_key_in_batch(key: str, batch: BatchDict):
    assert key in batch.keys(), (
        f"The following required key={key} not found in batch. "
        f"Batch keys={batch.keys()}"
    )


def _validate_keys_in_batch(keys: List[str], batch: BatchDict):
    for key in keys:
        _validate_key_in_batch(key, batch)
