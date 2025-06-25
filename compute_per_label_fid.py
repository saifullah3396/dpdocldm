import argparse
import glob
import logging
from typing import Any, Dict, List, Union
import warnings

import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from atria.core.data.data_modules.dataset_cacher.shard_list_datasets import (
    MsgpackListDataset,
)
from datadings.reader import MsgpackReader as MsgpackFileReader

import io

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


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
        print(self._data_files)
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
        return filtered_sample

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


def get_activations(dataloader, model, batch_elem_idx, dims=2048, device="cuda:0"):
    model.eval()
    pred_arr = np.empty((len(dataloader.dataset), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch[batch_elem_idx]
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


def compute_statistics_of_path(dataloader, model, batch_elem_idx, dims, device):
    act = get_activations(
        dataloader=dataloader,
        model=model,
        batch_elem_idx=batch_elem_idx,
        dims=dims,
        device=device,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid(dataloader, dims=2048, device: str = "cuda:0"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(
        dataloader=dataloader, model=model, batch_elem_idx=0, dims=dims, device=device
    )
    m2, s2 = compute_statistics_of_path(
        dataloader=dataloader, model=model, batch_elem_idx=1, dims=dims, device=device
    )
    return calculate_frechet_distance(m1, s1, m2, s2)


def arguments():
    parser = argparse.ArgumentParser(description="FVA arguments.")
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to generated images"
    )
    parser.add_argument(
        "--real_samples_dir",
        required=True,
        type=str,
        help="Path to real samples directory",
    )

    return parser.parse_args()


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, msgpack_files):
        data = []
        for file in msgpack_files:
            data


if __name__ == "__main__":
    args = arguments()
    all_samples_msgpacks = glob.glob(
        str(args.real_samples_dir / "**/samples/**/*samples.msgpack")
    )
    generated_dataset = MsgpackListDataset(all_samples_msgpacks)

    # all_metrics = []
    # with torch.no_grad():
    #     lpips_model = lpips.LPIPS(net="vgg").to(0)
    #     for run_dir in run_dirs:
    #         if "refined" in run_dir.name:
    #             continue
    #         if (run_dir / "metrics.csv").exists():
    #             prev_run = pd.read_csv(run_dir / "metrics.csv")
    #             if "mean_conf_score" in prev_run.columns:
    #                 logging.info(f"Skipping {run_dir} as metrics already computed")
    #                 all_metrics.append(pd.read_csv(run_dir / "metrics.csv"))
    #                 continue
    #             # continue

    #         results = []
    #         logging.info(f"Run dir: {run_dir}")
    #         info_dataset = CounterfactualInfoDataset(run_dir)
    #         dataloader_info = data.DataLoader(
    #             info_dataset,
    #             batch_size=256,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=8,
    #             pin_memory=True,
    #             collate_fn=lambda x: x,
    #         )

    #         logging.info("Computing counterfactual info")
    #         n_counterfactuals_found = 0
    #         mean_conf_score = 0
    #         total_samples = 0
    #         for info_batch in tqdm(dataloader_info):
    #             for info in info_batch:
    #                 target = info["target"]
    #                 cf_pred = info["cf pred"]
    #                 cf_inv_conf_score = info["cf_inv_conf_score"]
    #                 if target == cf_pred:
    #                     n_counterfactuals_found += 1
    #                     mean_conf_score += (
    #                         1 - cf_inv_conf_score
    #                     )  # take mean conf score only for flipped samples
    #                 total_samples += 1
    #         mean_conf_score /= n_counterfactuals_found
    #         flip_ratio = n_counterfactuals_found / total_samples
    #         logging.info(f"Flip ratio: {flip_ratio}")
    #         logging.info(f"Mean CF inv conf score: {mean_conf_score}")
    #         results.append(
    #             {
    #                 "run_dir": run_dir.name,
    #                 "flip_ratio": flip_ratio,
    #                 "mean_conf_score": mean_conf_score,
    #             }
    #         )

    #         dataset_real_samples_split_1 = CounterfactualDataset(
    #             real_samples_split_1_dir_path, run_dir
    #         )
    #         dataset_real_samples_split_2 = UnpairedCounterfactualDataset(
    #             real_samples_split_2_dir_path, run_dir
    #         )
    #         results.append(
    #             {
    #                 "run_dir": run_dir.name,
    #                 "size_info_dataset": len(info_dataset),
    #                 "size_dataset_real_samples_split_2": len(
    #                     dataset_real_samples_split_2
    #                 ),
    #                 "size_dataset_real_samples_split_1": len(
    #                     dataset_real_samples_split_1
    #                 ),
    #             }
    #         )
    #         dataloader_split_1 = data.DataLoader(
    #             dataset_real_samples_split_1,
    #             batch_size=256,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=8,
    #             pin_memory=True,
    #         )

    #         logging.info("Computing closeness metrics")
    #         for real, cf in tqdm(dataloader_split_1):
    #             real = real.to(0, dtype=torch.float)
    #             cf = cf.to(0, dtype=torch.float)
    #             bsz = real.shape[0]
    #             diff = real.view(bsz, -1) - cf.view(bsz, -1)
    #             l1_norm_sum = torch.norm(diff, p=1, dim=-1)
    #             l2_norm_sum = torch.norm(diff, p=2, dim=-1)
    #             l1_norm_mean = l1_norm_sum / diff.shape[-1]
    #             l2_norm_mean = l2_norm_sum / diff.shape[-1]
    #             lpips_loss = lpips_model(real, cf, normalize=True)

    #             for i in range(bsz):
    #                 results.append(
    #                     {
    #                         "run_dir": run_dir.name,
    #                         "l1_norm_sum": l1_norm_sum[i].item(),
    #                         "l2_norm_sum": l2_norm_sum[i].item(),
    #                         "l1_norm_mean": l1_norm_mean[i].item(),
    #                         "l2_norm_mean": l2_norm_mean[i].item(),
    #                         "lpips_loss": lpips_loss[i].item(),
    #                     }
    #                 )

    #         logging.info("Computing FID")
    #         fid = compute_fid(dataloader_split_1)
    #         dataloader_split_2 = data.DataLoader(
    #             dataset_real_samples_split_2,
    #             batch_size=256,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=8,
    #             pin_memory=True,
    #         )

    #         logging.info("Computing sFID")
    #         sfid = compute_fid(dataloader_split_2)
    #         results.append({"run_dir": run_dir.name, "fid": fid, "sfid": sfid})
    #         df = pd.DataFrame(results)
    #         df = df.groupby("run_dir").mean().reset_index()
    #         df.to_csv(run_dir / "metrics.csv", index=False)
    #         logging.info(df)
    #         all_metrics.append(pd.read_csv(run_dir / "metrics.csv"))

    # # Save results to a CSV file
    # results = pd.concat(all_metrics)
    # logging.info(results)
    # results.to_csv(experiment_dir / "metrics.csv", index=False)
