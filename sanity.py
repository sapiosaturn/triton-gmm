"""
just a sanity check script
"""
import torch
from triton_gmm import tgmm

num_groups = 4
in_features = 512
out_features = 256
total_tokens = 1024

activations = torch.randn(total_tokens, in_features, dtype=torch.bfloat16, device="cuda")
weights = torch.randn(num_groups, in_features, out_features, dtype=torch.bfloat16, device="cuda")
tokens_per_group = total_tokens // num_groups
offsets = torch.tensor(
    [tokens_per_group * (i + 1) for i in range(num_groups)],
    dtype=torch.int32,
    device="cuda"
)

torch_ref = torch._grouped_mm(activations, weights, offs=offsets)
triton_out = tgmm(activations, weights, offs=offsets)

diff = torch_ref - triton_out
max_diff = diff.abs().max().item()
mean_diff = diff.abs().mean().item()
max_rel_diff = (diff.abs() / (torch_ref.abs() + 1e-8)).max().item()

print(f"Max absolute difference: {max_diff:.6e}")
print(f"Mean absolute difference: {mean_diff:.6e}")
print(f"Max relative difference: {max_rel_diff:.6e}")
print(f"Shapes match: {torch_ref.shape == triton_out.shape}")
