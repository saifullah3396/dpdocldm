"""
Defines the GroupBatchSampler batch sampling strategy.
"""

from typing import List

from torch.utils.data.sampler import Sampler

from atria.data.batch_samplers.group_batch_sampler import GroupBatchSampler


class AspectRatioGroupBatchSampler(GroupBatchSampler):
    """
    Groups the input sample images based on their aspect ratio.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        group_factor: List[int],
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.group_factor = group_factor

    def __post_init__(self):
        from atria.data.batch_samplers.utilities import create_aspect_ratio_groups

        self.group_ids = create_aspect_ratio_groups(
            self.sampler.data_source,
            k=self.group_factor,
        )
