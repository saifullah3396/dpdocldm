import torch
from atria.core.utilities.logging import get_logger
from packaging.version import Version

if Version(torch.__version__) <= Version("1.7.0"):
    torch_outer = torch.ger
else:
    torch_outer = torch.outer


logger = get_logger(__name__)


def compute_fid(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    try:
        import numpy as np
    except ImportError:
        raise ModuleNotFoundError("fid_score requires numpy to be installed.")

    try:
        import scipy.linalg
    except ImportError:
        raise ModuleNotFoundError("fid_score requires scipy to be installed.")

    mu1, mu2 = mu1.cpu(), mu2.cpu()
    sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        try:
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
        except ValueError as e:
            logger.warning(e)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    if not np.isfinite(covmean).all():
        tr_covmean = np.sum(
            np.sqrt(((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps))
        )

    return float(
        diff.dot(diff).item()
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2 * tr_covmean
    )
