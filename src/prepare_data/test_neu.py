import os

import pytest

import neu


def tmp_fill(tmpdir_, classes, ext):
    for cls_ in classes.values():
        tmpdir_.join(cls_ + "_0" + ext)
    # add one more image
    tmpdir_.join(classes["crazing"] + "_1" + ext)
    return


@pytest.fixture()
def tmp_neu():
    def _create(tmpdir, filename):
        tmpneu = tmpdir.mkdir(filename)
        if filename == "NEU-CLS":
            tmp_fill(tmpneu, neu.CLASSES, ".png")
        elif filename == "NEU-DET":
            imgs = tmpneu.mkdir("IMAGES")
            tmp_fill(imgs, neu.CLASSES, ".png")
            anns = tmpneu.mkdir("ANNOTATIONS")
            tmp_fill(anns, neu.CLASSES, ".xml")
        else:
            raise ValueError("Not supported")
        return tmpneu

    return _create


@pytest.mark.parametrize("filename", ["NEU-CLS", "NEU-DET"])
def test_main(tmpdir, tmp_neu, filename) -> None:
    data_path = tmp_neu(tmpdir, filename)
    output_path = tmpdir.mkdir("output_path")
    neu.main(data_path, output_path, archived=False)
    assert len(os.listdir(output_path)) == len(neu.CLASSES), "failed to match number of classes in output"
    for p in output_path.visit():
        assert p.check(), "correct path was not created"
    return
