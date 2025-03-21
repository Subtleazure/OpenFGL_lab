import torch

S = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
print(S.mean().item())