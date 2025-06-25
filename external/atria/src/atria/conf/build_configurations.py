#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from atria.core.data.config import *  # noqa
from atria.core.metrics.classification.config import *  # noqa
from atria.core.metrics.layout.config import *  # noqa
from atria.core.metrics.qa.config import *  # noqa
from atria.core.metrics.token_classification.config import *  # noqa
from atria.core.models.config import *  # noqa
from atria.core.optimizers.config import *  # noqa
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.schedulers.config import *  # noqa
from atria.core.task_runners.config import *  # noqa
from atria.core.training.config import *  # noqa
from atria.data.batch_samplers.config import *  # noqa
from atria.data.data_transforms.config import *  # noqa
from atria.models.config import *  # noqa
from atria.models.model_configs import *  # noqa
from hydra.core.config_store import ConfigNode, ConfigStore
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def dump_configurations(root_dir: Path, d: dict):
    for k, value in d.items():
        if k in ["hydra", "_dummy_empty_config_.yaml"]:
            continue
        if isinstance(value, ConfigNode):
            logger.info(f"Dumping configuration: {value.group}/{value.name}")
            node_file_path = root_dir / value.group / value.name
            if not node_file_path.parent.exists():
                node_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(node_file_path, "w") as f:
                if value.package is not None:
                    f.write(f"# @package {value.package}\n")
                f.write(OmegaConf.to_yaml(value.node, sort_keys=False))
        elif isinstance(value, dict):
            dump_configurations(root_dir, value)


def build_configurations():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--configurations_dir",
        type=str,
        default=str(Path(__file__).parent),
    )
    args = args_parser.parse_args()
    cs: ConfigStore = AtriaModuleRegistry.build_module_configurations()

    root_package_dir = Path(args.configurations_dir)
    logger.info(f"Building atria configurations.")
    dump_configurations(root_package_dir, cs.repo)


if __name__ == "__main__":
    build_configurations()
