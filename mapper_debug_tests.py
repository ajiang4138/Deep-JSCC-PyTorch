import torch

from constellation import (MICLayer, average_symbol_power, map_to_mic_codebook,
                           pair_channels_to_symbols,
                           unpair_symbols_to_channels)
from model import DeepJSCC


def test_pair_unpair_roundtrip(device='cpu'):
    x = torch.randn(2, 8, 6, 6, device=device)
    paired = pair_channels_to_symbols(x)
    restored = unpair_symbols_to_channels(paired)
    assert restored.shape == x.shape
    assert torch.allclose(x, restored, atol=1e-6), 'pair/unpair roundtrip failed'


def test_gradient_flow_to_codebook(device='cpu'):
    mapper = MICLayer(
        constellation_size=16,
        clip_value=2.0,
        temperature=0.2,
        hard_forward=True,
        train_mode='hard_forward_soft_backward',
        power_constraint_mode='codebook',
    ).to(device)

    z = torch.randn(2, 8, 8, 8, device=device, requires_grad=True)
    mapped = mapper(z)
    loss = mapped.pow(2).mean()
    loss.backward()

    assert z.grad is not None, 'No gradient to encoder output surrogate'
    assert mapper.codebook.grad is not None, 'No gradient to MIC codebook'


def test_mapper_none_compatibility(device='cpu'):
    torch.manual_seed(0)
    model_legacy = DeepJSCC(c=4, channel_type='AWGN', snr=None).to(device).eval()
    torch.manual_seed(0)
    model_none = DeepJSCC(c=4, channel_type='AWGN', snr=None, mapper_type='none').to(device).eval()

    model_none.load_state_dict(model_legacy.state_dict())

    x = torch.randn(1, 3, 32, 32, device=device)
    y_legacy = model_legacy(x)
    y_none = model_none(x)

    assert y_legacy.shape == y_none.shape
    assert torch.allclose(y_legacy, y_none, atol=1e-6), 'mapper_type=none path changed behavior'


def test_hard_deploy_outputs_codebook_points(device='cpu'):
    mapper = MICLayer(
        constellation_size=16,
        clip_value=2.0,
        temperature=0.2,
        hard_forward=True,
        train_mode='hard_forward_soft_backward',
        power_constraint_mode='codebook',
    ).to(device)
    mapper.set_deploy_mode(True)

    z = torch.randn(1, 8, 4, 4, device=device)
    mapped, indices = mapper(z, return_indices=True)

    paired = pair_channels_to_symbols(mapped)
    flat_symbols = paired.view(-1, 2)
    codebook = mapper.get_effective_codebook()
    expected = codebook.index_select(0, indices.view(-1))

    assert torch.allclose(flat_symbols, expected, atol=1e-6), 'Hard deploy output not on MIC codebook points'

    mapped_ext, idx_ext = map_to_mic_codebook(z, codebook=codebook, clip_value=2.0, power_constraint_mode='codebook')
    paired_ext = pair_channels_to_symbols(mapped_ext).view(-1, 2)
    expected_ext = codebook.index_select(0, idx_ext.view(-1))
    assert torch.allclose(paired_ext, expected_ext, atol=1e-6), 'Standalone map_to_mic_codebook mismatch'


def test_codebook_normalization(device='cpu'):
    mapper = MICLayer(
        constellation_size=16,
        clip_value=2.0,
        temperature=0.2,
        hard_forward=True,
        train_mode='hard_forward_soft_backward',
        power_constraint_mode='codebook',
    ).to(device)

    with torch.no_grad():
        mapper.codebook.mul_(5.0)

    effective = mapper.get_effective_codebook()
    pwr = average_symbol_power(effective).item()
    assert abs(pwr - 1.0) < 1e-3, 'Codebook normalization failed to enforce unit power'


def test_cpu_gpu_eval():
    test_mapper_none_compatibility(device='cpu')
    test_pair_unpair_roundtrip(device='cpu')
    test_gradient_flow_to_codebook(device='cpu')
    test_hard_deploy_outputs_codebook_points(device='cpu')
    test_codebook_normalization(device='cpu')

    if torch.cuda.is_available():
        test_mapper_none_compatibility(device='cuda')
        test_pair_unpair_roundtrip(device='cuda')
        test_gradient_flow_to_codebook(device='cuda')
        test_hard_deploy_outputs_codebook_points(device='cuda')
        test_codebook_normalization(device='cuda')


if __name__ == '__main__':
    test_cpu_gpu_eval()
    print('All mapper sanity checks passed.')
