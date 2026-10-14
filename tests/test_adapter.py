import numpy as np
import pytest
import torch

from cloudlift.model import adapter
from cloudlift.model.uncrtaints_wrapper import build_backbone, load, save_cloudlift_weights


def test_stack_inputs_layout():
    cloudy = np.full((3, 8, 8), 0.5, np.float32)
    sar = np.full((2, 8, 8), 0.2, np.float32)
    ref = np.full((3, 8, 8), 0.1, np.float32)
    x, d = adapter.stack_inputs(cloudy, sar, [ref], input_t=2, ref_gap_days=40)
    assert x.shape == (2, 5, 8, 8)
    assert np.allclose(x[:, :2], 0.2)  # SAR first at every step
    assert np.allclose(x[0, 2:], 0.1) and np.allclose(x[1, 2:], 0.5)  # reference then cloudy
    assert d.tolist() == [0.0, 40.0]
    x3, d3 = adapter.stack_inputs(cloudy, sar, [ref], input_t=3, ref_gap_days=40)
    assert x3.shape == (3, 5, 8, 8) and d3.tolist() == [0.0, 0.0, 40.0]
    x1, _ = adapter.stack_inputs(cloudy, sar, [ref], input_t=1)
    assert x1.shape == (1, 5, 8, 8)


def test_surgery_copies_pretrained_filters():
    torch.manual_seed(0)
    bb = build_backbone("tiny", 15, 26)
    w_in = adapter._first_conv(bb.in_conv).weight.detach().clone()
    w_out = adapter._last_conv(bb.out_conv).weight.detach().clone()
    adapter.adapt_input_conv(bb)
    adapter.adapt_output_head(bb, "diag")
    new_in = adapter._first_conv(bb.in_conv).weight
    new_out = adapter._last_conv(bb.out_conv).weight
    assert new_in.shape[1] == 5 and new_out.shape[0] == 6
    assert torch.equal(new_in[:, :2], w_in[:, :2])
    assert torch.equal(new_in[:, 2:], w_in[:, [4, 5, 9]])  # S2 B3, B4, B8 after 2 SAR channels
    assert torch.equal(new_out, w_out[[2, 3, 7, 15, 16, 20]])
    assert (bb.mean_idx, bb.vars_idx) == (3, 6)


def test_upstream_checkpoint_load(tmp_path):
    bb = build_backbone("full", 15, 26)
    p = tmp_path / "model.pth.tar"
    torch.save({"state_dict_G": bb.state_dict()}, p)
    net = load(p)
    assert net.trained and net.size == "full" and net.weights_source.startswith("uncrtaints-pretrained")
    mu, sg = net.eval()(torch.rand(1, 2, 5, 64, 64), torch.tensor([[0.0, 30.0]]))
    assert mu.shape == (1, 3, 64, 64) and sg.shape == (1, 3, 64, 64)


def test_wrapper_shapes_and_untrained_flag():
    net = load(None, size="tiny").eval()
    assert not net.trained and net.weights_hash.startswith("untrained")
    mu, sg = net(torch.rand(2, 2, 5, 64, 64), torch.tensor([[0.0, 30.0]] * 2))
    assert mu.shape == (2, 3, 64, 64) and sg.shape == (2, 3, 64, 64)
    assert (sg > 0).all() and (mu >= 0).all() and (mu <= 1).all()
    with pytest.raises(ValueError):
        net(torch.rand(1, 2, 4, 64, 64), torch.zeros(1, 2))


def test_iso_covmode():
    net = load(None, size="tiny", covmode="iso").eval()
    _, sg = net(torch.rand(1, 2, 5, 64, 64), torch.tensor([[0.0, 30.0]]))
    assert sg.shape == (1, 1, 64, 64)


def test_cloudlift_checkpoint_roundtrip(tmp_path):
    a = load(None, size="tiny", seed=4).eval()
    p = save_cloudlift_weights(a, tmp_path / "w.pt", trained_on="unit-test")
    b = load(p).eval()
    x, d = torch.rand(1, 2, 5, 64, 64), torch.tensor([[0.0, 30.0]])
    assert torch.equal(a(x, d)[0], b(x, d)[0]) and b.trained


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_full_forward_fits_6gb_amp():
    net = load(None, size="full", gradient_checkpointing=True).cuda()
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(1, 2, 5, 256, 256, device="cuda")
    d = torch.tensor([[0.0, 30.0]], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        mu, sg = net(x, d)
        (mu.float().mean() + sg.float().mean()).backward()
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert peak < 5.0, f"peak {peak:.2f} GiB"
