import re
from pathlib import Path


def get_score(s: str) -> float:
    """Gets the criterion score from .ckpt formated with ModelCheckpoint

    Parameters
    ----------
    s : str
        Assumption is the last number is the desired number carved in .ckpt

    Returns
    -------
    float
        The criterion float
    """
    return float(re.findall(r"(\d+.\d+).ckpt", s)[0])


def main(path: str, op: str) -> str:
    """Finds the best ckpt path

    Parameters
    ----------
    path : str
        ckpt path
    op : str
        "max" (for mAP for example) or "min" (for loss)

    Returns
    -------
    str
        A ckpt path
    """
    ckpts = list(map(str, Path(path).glob("*.ckpt")))
    if not len(ckpts):
        return

    ckpt_score_dict = {ckpt: get_score(ckpt) for ckpt in ckpts}
    op = max if op == "max" else min
    out = op(ckpt_score_dict, key=ckpt_score_dict.get)
    print(out)  # need to flush for bash
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("provide checkpoint path and op either max or min")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
