import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf


def _flatten_hydra_config(cfg : DictConfig, out_dict : dict, prefix : str = ""):
    new_prefix = prefix + "." if len(prefix) > 0 else ""
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            _flatten_hydra_config(value, out_dict, new_prefix + key)
        else:
            out_dict[new_prefix + key] = value

def flatten_hydra_config(cfg : DictConfig) -> dict:
    out_dict = {}
    _flatten_hydra_config(cfg, out_dict)
    return out_dict

def wandb_kwargs_via_cfg(cfg : DictConfig, use_group_name: bool = True) -> dict:
    """Converts a Hydra config into a flat dict for wandb logging."""

    wandb_kwargs = flatten_hydra_config(cfg)
    
    return wandb_kwargs

