#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from atria.core.registry.module_registry import AtriaModuleRegistry
from hydra.core.config_store import ConfigNode, ConfigStore
from omegaconf import OmegaConf

from dp_diffusion.config import *  # noqa
from dp_diffusion.models.model_configs import *  # noqa

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
    logger.info(f"Building dp_diffusion configurations.")
    dump_configurations(root_package_dir, cs.repo)


if __name__ == "__main__":
    build_configurations()
