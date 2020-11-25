from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent.resolve()

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

dev_dependencies = ["pre-commit", "mypy==0.781", "black==20.8b1", "nbstripout==0.3.7", "black-nb==0.3.0"]
test_dependencies = ["pytest>=6.0"]
doc_dependencies = ["portray>=1.4.0"]

setup(
    name="sagemaker_defect_detection",
    version="0.1",
    description="Detect Defects in Products from their Images using Amazon SageMaker ",
    long_description=long_description,
    author="Ehsan M. Kermani",
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=find_packages("src", exclude=["tests", "tests/*"]),
    install_requires=open(str(ROOT / "requirements.txt"), "r").read(),
    extras_require={"dev": dev_dependencies, "test": test_dependencies, "doc": doc_dependencies},
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
