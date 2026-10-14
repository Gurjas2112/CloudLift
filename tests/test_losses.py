import torch

from cloudlift.config import LossCfg
from cloudlift.model.losses import CombinedLoss, gaussian_nll, sam_loss, spectral_angle, ssim_loss


def test_sam_identical_zero_and_scale_invariant():
    x = torch.rand(2, 3, 16, 16) + 0.05
    assert spectral_angle(x, x).max() < 1e-4
    y = torch.rand(2, 3, 16, 16) + 0.05
    a = spectral_angle(x, y)
    assert torch.allclose(a, spectral_angle(3.7 * x, 0.2 * y), atol=1e-5)


def test_sam_known_angle():
    x = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    y = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    assert torch.isclose(spectral_angle(x, y), torch.tensor(torch.pi / 2), atol=1e-6).all()


def test_ssim_identical():
    x = torch.rand(1, 3, 32, 32)
    assert ssim_loss(x, x) < 1e-5


def test_nll_minimised_at_true_sigma():
    torch.manual_seed(0)
    y = torch.zeros(1, 3, 64, 64)
    mu = torch.randn_like(y) * 0.1
    vals = {s: gaussian_nll(mu, torch.full_like(y, s), y).item() for s in (0.03, 0.1, 0.3)}
    assert vals[0.1] < vals[0.03] and vals[0.1] < vals[0.3]


def test_combined_backward_and_masking():
    recon = torch.rand(2, 3, 32, 32, requires_grad=True)
    sigma = (torch.rand(2, 3, 32, 32) * 0.1 + 0.01).requires_grad_()
    target = torch.rand(2, 3, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss, parts = CombinedLoss(LossCfg())(recon, sigma, target, mask)
    loss.backward()
    assert torch.isfinite(recon.grad).all() and torch.isfinite(sigma.grad).all()
    assert set(parts) == {"l1", "ssim", "sam", "nll", "total"}


def test_sam_gradient_finite_at_identity():
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    sam_loss(x, x.detach()).backward()
    assert torch.isfinite(x.grad).all()
