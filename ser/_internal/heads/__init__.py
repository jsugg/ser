"""Internal classifier head helpers."""

from ser._internal.heads.torch_head import build_torch_mlp_head, forward_torch_head

__all__ = ["build_torch_mlp_head", "forward_torch_head"]
