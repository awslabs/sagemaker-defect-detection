"""
Dependencies: unzip unrar
python -m pip install patool pyunpack
"""

from pathlib import Path
import shutil
import re
import os

try:
    from pyunpack import Archive
except ModuleNotFoundError:
    print("installing the dependencies `patool` and `pyunpack` for unzipping the data")
    import subprocess

    subprocess.run("python -m pip install patool==1.12 pyunpack==0.2.1 -q", shell=True)
    from pyunpack import Archive

CLASSES = {
    "crazing": "Cr",
    "inclusion": "In",
    "pitted_surface": "PS",
    "patches": "Pa",
    "rolled-in_scale": "RS",
    "scratches": "Sc",
}


def unpack(path: str) -> None:
    path = Path(path)
    Archive(str(path)).extractall(str(path.parent))
    return


def cp_class_images(data_path: Path, class_name: str, class_path_dest: Path) -> None:
    lst = list(data_path.rglob(f"{class_name}_*"))
    for img_file in lst:
        shutil.copy2(str(img_file), str(class_path_dest / img_file.name))

    assert len(lst) == len(list(class_path_dest.glob("*")))
    return


def cp_image_annotation(data_path: Path, class_name: str, image_path_dest: Path, annotation_path_dest: Path) -> None:
    img_lst = sorted(list((data_path / "IMAGES").rglob(f"{class_name}_*")))
    ann_lst = sorted(list((data_path / "ANNOTATIONS").rglob(f"{class_name}_*")))
    assert len(img_lst) == len(
        ann_lst
    ), f"images count {len(img_lst)} does not match with annotations count {len(ann_lst)} for class {class_name}"
    for (img_file, ann_file) in zip(img_lst, ann_lst):
        shutil.copy2(str(img_file), str(image_path_dest / img_file.name))
        shutil.copy2(str(ann_file), str(annotation_path_dest / ann_file.name))

    assert len(list(image_path_dest.glob("*"))) == len(list(annotation_path_dest.glob("*")))
    return


def main(data_path: str, output_path: str, archived: bool = True) -> None:
    """
    Data preparation

    Parameters
    ----------
    data_path : str
        Raw data path
    output_path : str
        Output data path
    archived: bool
        Whether the file is archived or not (for testing)

    Raises
    ------
    ValueError
        If the packed data file is different from NEU-CLS or NEU-DET
    """
    data_path = Path(data_path)
    if archived:
        unpack(data_path)

    data_path = data_path.parent / re.search(r"^[^.]*", str(data_path.name)).group(0)
    try:
        os.remove(str(data_path / "Thumbs.db"))
    except FileNotFoundError:
        print(f"Thumbs.db is not found. Continuing ...")
        pass
    except Exception as e:
        print(f"{e}: Unknown error!")
        raise e

    output_path = Path(output_path)
    if data_path.name == "NEU-CLS":
        for cls_ in CLASSES.values():
            cls_path = output_path / cls_
            cls_path.mkdir(exist_ok=True)
            cp_class_images(data_path, cls_, cls_path)
    elif data_path.name == "NEU-DET":
        for cls_ in CLASSES:
            cls_path = output_path / CLASSES[cls_]
            image_path = cls_path / "images"
            image_path.mkdir(parents=True, exist_ok=True)
            annotation_path = cls_path / "annotations"
            annotation_path.mkdir(exist_ok=True)
            cp_image_annotation(data_path, cls_, image_path, annotation_path)
    else:
        raise ValueError(f"Unknown data. Choose between `NEU-CLS` and `NEU-DET`. Given {data_path.name}")

    return


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Provide `data_path` and `output_path`")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
    print("Done")
