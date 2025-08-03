import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import torch

_logger = logging.getLogger(__name__)


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict
    # if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def init_device(args):
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(device)
    return


def load_state_dict(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ""
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model" in checkpoint:
                state_dict_key = "model"
        state_dict = clean_state_dict(
            checkpoint[state_dict_key] if state_dict_key else checkpoint
        )
        _logger.info(
            "Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path)
        )
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    remap: bool = False,
    filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return

    state_dict = load_state_dict(checkpoint_path, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def remap_state_dict(
    state_dict: Dict[str, Any], model: torch.nn.Module, allow_reshape: bool = True
):
    """remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created
    with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert (
            va.numel() == vb.numel()
        ), f"Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed."
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert (
                    False
                ), f"Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed."
        out_dict[ka] = vb
    return out_dict
