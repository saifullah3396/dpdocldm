# %%
import io
import glob
from typing import Any, Dict, List, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from datadings.reader import MsgpackReader as MsgpackFileReader
import glob
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from datadings.reader import MsgpackReader as MsgpackFileReader

import io


class MsgpackShardListDataset(Dataset):
    """
    A mix of torch dataset and huggingface dataset info backed by a msgpack file or a list of msgpack files.
    This dataset is loaded by the MsgpackBuilder.
    """

    def __init__(
        self,
        files: List[str],
        transforms: transforms.Compose,
    ) -> None:
        import numpy as np

        self._data_files = sorted(files)
        self._data = [MsgpackFileReader(f) for f in self._data_files]
        self._transformations = transforms

        self._cumulative_sizes: List[int] = []
        self._total_size: int = 0

        for data in self._data:
            self._total_size += len(data)
            self._cumulative_sizes.append(self._total_size)
            data._close()
        self._cumulative_sizes = np.array(self._cumulative_sizes)
        self._features_metadata = None
        self._apply_postprocessing = True

    def get_shard(self, index: int) -> Union[MsgpackFileReader, int, str]:
        import numpy as np

        shard_index = np.searchsorted(self._cumulative_sizes, index, side="right")

        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - self._cumulative_sizes[shard_index - 1]

        shard = self._data[shard_index]
        url = self._data_files[shard_index]
        return shard, inner_index, url

    def __getitem__(self, index: int) -> Dict[str, Any]:
        shard, inner_idx, url = self.get_shard(index)
        sample = shard[inner_idx]

        sample["__index__"] = index
        sample["__shard__"] = url
        sample["__shardindex__"] = inner_idx

        filtered_sample = {}
        filtered_sample["image"] = self._transformations(
            Image.open(io.BytesIO(sample["image.mp"]["bytes"])).convert("RGB")
        )
        filtered_sample["label"] = sample["label.mp"]
        return filtered_sample["image"]

    def __len__(self) -> int:
        """
        Get the total number of elements in the dataset.

        Returns:
            int: Total number of elements.
        """
        return self._total_size

    def __repr__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        return f"Dataset({{\n    features: {list(self._info.features.keys())},\n    num_rows: {self._total_size}\n}})"


def _create_random_subset(dataset: Dataset, max_samples: int):
    import torch
    from torch.utils.data import Subset

    max_samples = min(max_samples, len(dataset))
    dataset = Subset(
        dataset,
        torch.randperm(len(dataset))[:max_samples],
    )
    return dataset


def get_activations(dataloader, model, dims=2048, device="cuda:0"):
    model.eval()
    pred_arr = np.empty((len(dataloader.dataset), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        # _plot_tensors(batch, "batch")
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr


def compute_statistics_of_path(dataloader, model, dims, device):
    act = get_activations(
        dataloader=dataloader,
        model=model,
        dims=dims,
        device=device,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class FIDEvaluator:
    def __init__(self, dataloader_real, dims=2048, device="cuda:0"):
        self.dims = dims
        self.device = device
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to("cuda:0")
        self.m1, self.s1 = compute_statistics_of_path(
            dataloader=dataloader_real, model=self.model, dims=dims, device=device
        )

    def compute_fid(self, dataloader_generated):
        m2, s2 = compute_statistics_of_path(
            dataloader=dataloader_generated,
            model=self.model,
            dims=self.dims,
            device=self.device,
        )
        return calculate_frechet_distance(self.m1, self.s1, m2, s2)


def compute_fid_for_dir(
    generated_samples_dirs, real_samples_dir, num_fid_samples=50000
):
    all_real_samples = glob.glob(str(real_samples_dir / "256x256-train-*.msgpack"))
    all_real_samples = [x for x in all_real_samples if "features" not in x]
    real_dataset = MsgpackShardListDataset(
        all_real_samples, transforms.Compose([transforms.ToTensor()])
    )
    print("Length of real dataset", len(real_dataset))
    real_dataset = _create_random_subset(real_dataset, num_fid_samples)
    print("Length of real dataset", len(real_dataset))
    dataloader_real = torch.utils.data.DataLoader(
        real_dataset, batch_size=512, shuffle=True, num_workers=8
    )
    fid_evaluator = FIDEvaluator(dataloader_real, dims=2048, device="cuda:0")
    fids = []
    for generated_samples_dir in generated_samples_dirs:
        all_generated_samples = glob.glob(
            str(generated_samples_dir / "**/samples/**/*samples.msgpack")
        )
        generated_dataset = MsgpackShardListDataset(
            all_generated_samples, transforms.Compose([transforms.ToTensor()])
        )
        print("Length of generated dataset", len(generated_dataset))
        generated_dataset = _create_random_subset(generated_dataset, num_fid_samples)
        print("Length of generated dataset", len(generated_dataset))
        dataloader_generated = torch.utils.data.DataLoader(
            generated_dataset, batch_size=512, shuffle=True, num_workers=8
        )
        fid = fid_evaluator.compute_fid(dataloader_generated)
        fids.append(fid)

        print(f"{generated_samples_dir.name}: fid={fid}")
    return fids


real_samples_dir = Path(
    "/ds-sds/documents/rvlcdip/.atria/msgpack/rvlcdip/images_with_text-f95ddecf3a1eaf13/0.0.0/"
)
base_path = Path(
    "/netscratch/saifullah/synced_projects/dp_diffusion_v1/output/atria_trainer/RvlCdip/"
)
generated_samples_dirs = [
    "rvlcdip_dp_aug_klf4_cfg_per_label",
    "rvlcdip_dp_aug_sbv1.4_cfg_per_label",
    "rvlcdip_dp_promise_aug_klf4_cfg_per_label",
    "rvlcdip_dp_promise_aug_sbv1.4_cfg_per_label",
    "rvlcdip_dp_unaug_klf4_cfg_per_label",
    "rvlcdip_dp_unaug_sbv1.4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_klf4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_sbv1.4_cfg_per_label",
    "rvlcdip_dp_unaug_class_cond_klf4_cfg_per_label",
    "rvlcdip_dp_unaug_class_cond_sbv1.4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_class_cond_klf4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_class_cond_sbv1.4_cfg_per_label",
    "rvlcdip_dp_unaug_layout_cond_klf4_cfg_per_label",
    "rvlcdip_dp_unaug_layout_cond_sbv1.4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_layout_cond_klf4_cfg_per_label",
    "rvlcdip_dp_promise_unaug_layout_cond_sbv1.4_cfg_per_label",
    "rvlcdip_final_dp_unaug_layout_cond_klf4_cfg_per_label_eps_1",
    "rvlcdip_final_dp_unaug_layout_cond_klf4_cfg_per_label_eps_10",
    "rvlcdip_final_dp_unaug_layout_cond_klf4_cfg_per_label_eps_5",
    "rvlcdip_final_dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_1",
    "rvlcdip_final_dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_10",
    "rvlcdip_final_dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_5",
]

generated_samples_dirs = [base_path / x for x in generated_samples_dirs]

with torch.no_grad():
    fids = compute_fid_for_dir(generated_samples_dirs, real_samples_dir)
