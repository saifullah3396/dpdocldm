import dataclasses
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import datasets
from datasets import GeneratorBasedBuilder

from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class AtriaHuggingfaceDatasetConfig(datasets.BuilderConfig):
    """
    BuilderConfig for HuggingfaceDataset.

    Attributes:
        data_url (str): URL to download the dataset from.
        homepage (str): Homepage of the dataset.
        citation (str): Citation information for the dataset.
        license (str): License information for the dataset.
    """

    data_url: str = None
    homepage: str = None
    citation: str = None
    license: str = None


class AtriaHuggingfaceDataset(GeneratorBasedBuilder, ABC):
    """
    Abstract base class for Huggingface datasets.

    This class inherits from GeneratorBasedBuilder and ABC (Abstract Base Class).
    It provides a template for creating datasets that can be used with the Huggingface datasets library.

    Attributes:
        BUILDER_CONFIGS (list): List of dataset configurations.
    """

    def _relative_data_dir(self, with_version=True, with_hash=False) -> str:
        """
        Returns the relative data directory.

        Args:
            with_version (bool): Whether to include the version in the directory path.
            with_hash (bool): Whether to include the hash in the directory path.

        Returns:
            str: The relative data directory path.
        """
        return super()._relative_data_dir(
            with_version=with_version, with_hash=False
        )  # we do not add hash to keep it simple

    def _prepare_data_dir(self, dl_manager: datasets.DownloadManager):
        if self.config.data_url is None and self.config.data_dir is None:
            raise ValueError("You must provide either data_url or data_dir")

        if self.config.data_dir is not None:
            dl_manager.download_config.cache_dir = self.config.data_dir
        else:
            self.config.data_dir = dl_manager.download_config.cache_dir

        if self.config.data_url is not None:
            urls = []
            url_output_paths = []
            if isinstance(self.config.data_url, dict):
                urls = list(self.config.data_url.values())
                url_output_paths = list(self.config.data_url.keys())
            elif isinstance(self.config.data_url, list):
                urls = self.config.data_url
                url_output_paths = [
                    Path(url)
                    .name.replace(".tar.gz", "")
                    .replace(".zip", "")
                    .replace(".tar", "")
                    for url in urls
                ]
            else:
                urls = [self.config.data_url]
                url_output_paths = [
                    Path(urls[0])
                    .name.replace(".tar.gz", "")
                    .replace(".zip", "")
                    .replace(".tar", "")
                ]

            downloaded_urls = {}
            remaining_urls = {}
            for url, url_output_path in zip(urls, url_output_paths):
                download_path = (
                    Path(dl_manager.download_config.cache_dir) / url_output_path
                )
                if download_path.exists():
                    downloaded_urls[str(download_path)] = url
                else:
                    remaining_urls[str(download_path)] = url
            if len(remaining_urls) > 0:
                remaining_urls = dl_manager.download_and_extract(remaining_urls)
                for k, v in remaining_urls.items():
                    if not Path(k).parent.exists():
                        Path(k).parent.mkdir(parents=True, exist_ok=True)
                    Path(v).rename(k)
                downloaded_urls.update(remaining_urls)

            # remove download cache files
            for file in os.listdir(dl_manager.download_config.cache_dir):
                if file.endswith(".lock"):
                    os.remove(os.path.join(dl_manager.download_config.cache_dir, file))
                if file.endswith(".incomplete"):
                    os.remove(os.path.join(dl_manager.download_config.cache_dir, file))
                if file.endswith(".json"):
                    os.remove(os.path.join(dl_manager.download_config.cache_dir, file))

            extract_dir = Path(dl_manager.download_config.cache_dir) / "extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

            return {Path(k).name: v for k, v in downloaded_urls.items()}
        else:
            return {}

    def _info(self):
        """
        Returns the dataset information.

        Returns:
            AtriaDatasetInfo: The dataset information.
        """
        return AtriaDatasetInfo(
            description=self.config.description,
            citation=self.config.citation,
            homepage=self.config.homepage,
            license=self.config.license,
            features=self._dataset_features(),
            version=self.config.version,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.

        Args:
            dl_manager (datasets.DownloadManager): The download manager to download and extract data.

        Returns:
            list: List of SplitGenerators.
        """
        downloaded_urls = self._prepare_data_dir(dl_manager)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"downloaded_urls": downloaded_urls, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"downloaded_urls": downloaded_urls, "split": "test"},
            ),
        ]

    @abstractmethod
    def _dataset_features(self):
        """
        Abstract method to define dataset features.

        Subclasses must implement this method to specify the features of the dataset.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement _dataset_features()")

    @abstractmethod
    def _generate_examples(self, *args, **kwargs):
        """
        Abstract method to generate examples.

        Subclasses must implement this method to generate examples for the dataset.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError()
