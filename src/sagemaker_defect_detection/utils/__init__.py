from typing import Optional, Union
from pathlib import Path
import tarfile
import logging
from logging.config import fileConfig

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_logger(config_path: str) -> logging.Logger:
    fileConfig(config_path, disable_existing_loggers=False)
    logger = logging.getLogger()
    return logger


def str2bool(flag: Union[str, bool]) -> bool:
    if not isinstance(flag, bool):
        if flag.lower() == "false":
            flag = False
        elif flag.lower() == "true":
            flag = True
        else:
            raise ValueError("Wrong boolean argument!")
    return flag


def freeze(m: nn.Module) -> None:
    assert isinstance(m, nn.Module), "freeze only is applied to modules"
    for param in m.parameters():
        param.requires_grad = False

    return


def load_checkpoint(model: nn.Module, path: str, prefix: Optional[str]) -> nn.Module:
    path = Path(path)
    logger.info(f"path: {path}")
    if path.is_dir():
        path_str = str(list(path.rglob("*.ckpt"))[0])
    else:
        path_str = str(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(path_str, map_location=torch.device(device))["state_dict"]
    if prefix is not None:
        if prefix[-1] != ".":
            prefix += "."

        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    model.load_state_dict(state_dict, strict=True)
    return model
