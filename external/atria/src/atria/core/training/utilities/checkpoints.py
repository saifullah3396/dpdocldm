from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def find_checkpoint_file(
    filename, checkpoint_dir: str, load_best: bool = False, resume=True, quiet=False
):
    import glob
    import os
    from pathlib import Path

    if not checkpoint_dir.exists():
        return

    if filename is not None:
        if Path(filename).exists():
            return Path(filename)
        elif Path(checkpoint_dir / filename).exists():
            return Path(checkpoint_dir / filename)
        else:
            logger.warning(
                f"User provided checkpoint file filename={filename} not found."
            )

    list_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if len(list_checkpoints) > 0:
        if not load_best:
            list_checkpoints = [c for c in list_checkpoints if "best" not in c]
        else:
            list_checkpoints = [c for c in list_checkpoints if "best" in c]

        if len(list_checkpoints) > 0:
            latest_checkpoint = max(list_checkpoints, key=os.path.getctime)
            if resume:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, resuming training from {latest_checkpoint}. To avoid this behavior, change "
                        "the `output_dir` or add `overwrite_output_dir` to train from scratch."
                    )
            else:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, testing model using checkpoint {latest_checkpoint}."
                    )
            return latest_checkpoint


def find_resume_checkpoint(
    resume_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    return find_checkpoint_file(
        filename=resume_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=True,
    )


def find_test_checkpoint(
    test_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    return find_checkpoint_file(
        filename=test_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=False,
    )
