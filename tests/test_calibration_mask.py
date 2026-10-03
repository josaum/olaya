import torch

from olaya.calibration import fit_temperature


def test_calibration_never_assigns_mass_to_padded_candidates():
    logits = torch.tensor([[2.0, 1.0, -1e4], [1.0, 2.0, -1e4]])
    targets = torch.tensor([0, 1])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    _, before, after = fit_temperature(logits, targets, init_temp=1e8, max_iter=1, candidate_mask=mask)
    assert before.accuracy == 1.0
    assert abs(after.avg_confidence - 0.5) < 1e-6
    assert abs(after.nll - torch.log(torch.tensor(2.0)).item()) < 1e-6
