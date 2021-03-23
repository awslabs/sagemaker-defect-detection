#!/bin/bash
set -euxo pipefail

NOW=$(date +"%x %r %Z")
echo "Time: $NOW"

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: bash ./scripts/build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

REGION=$3
SOURCE_REGION="us-west-2"
echo "Region: $REGION, source region: $SOURCE_REGION"
BASE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
echo "Base dir: $BASE_DIR"

rm -rf build

# build python package
echo "Python build and package"
python -m pip install --upgrade pip
python -m pip install --upgrade wheel setuptools
python setup.py build sdist bdist_wheel

find . | grep -E "(__pycache__|\.pyc|\.pyo$|\.egg*|\lightning_logs)" | xargs rm -rf

echo "Add requirements for SageMaker"
cp requirements.txt build/lib/
cd build/lib || exit
mv sagemaker_defect_detection/{classifier.py,detector.py} .
touch source_dir.tar.gz
tar --exclude=source_dir.tar.gz -czvf source_dir.tar.gz .
echo "Only keep source_dir.tar.gz for SageMaker"
find . ! -name "source_dir.tar.gz" -type f -exec rm -r {} +
rm -rf sagemaker_defect_detection

cd - || exit

mv dist build

# add notebooks to build
echo "Prepare notebooks and add to build"
cp -r notebooks build
rm -rf build/notebooks/*neu* # remove local datasets for build
for nb in build/notebooks/*.ipynb; do
    python "$BASE_DIR"/scripts/set_kernelspec.py --notebook "$nb" --display-name "Python 3 (PyTorch JumpStart)" --kernel "HUB_1P_IMAGE"
done

echo "Copy src to build"
cp -r src build

# add solution assistant
echo "Solution assistant lambda function"
cd cloudformation/solution-assistant/ || exit
python -m pip install -r requirements.txt -t ./src/site-packages

cd - || exit

echo "Clean up pyc files, needed to avoid security issues. See: https://blog.jse.li/posts/pyc/"
find cloudformation | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
cp -r cloudformation/solution-assistant build/
cd build/solution-assistant/src || exit
zip -q -r9 "$BASE_DIR"/build/solution-assistant.zip -- *

cd - || exit
rm -rf build/solution-assistant

if [ -z "$4" ] || [ "$4" == 'mainline' ]; then
    s3_prefix="s3://$2-$3/$1"
else
    s3_prefix="s3://$2-$3/$1-$4"
fi

# cleanup and copy the build artifacts
echo "Removing the existing objects under $s3_prefix"
aws s3 rm --recursive "$s3_prefix" --region "$REGION"
echo "Copying new objects to $s3_prefix"
aws s3 sync . "$s3_prefix" --delete --region "$REGION" \
    --exclude ".git/*" \
    --exclude ".vscode/*" \
    --exclude ".mypy_cache/*" \
    --exclude "logs/*" \
    --exclude "stack_outputs.json" \
    --exclude "src/sagemaker_defect_detection/lightning_logs/*" \
    --exclude "notebooks/*neu*/*"

echo "Copying solution artifacts"
aws s3 cp "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/demo/model.tar.gz" "$s3_prefix"/demo/model.tar.gz --source-region "$SOURCE_REGION"

mkdir -p build/pretrained/
aws s3 cp "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/pretrained/model.tar.gz" build/pretrained --source-region "$SOURCE_REGION" &&
    cd build/pretrained/ && tar -xf model.tar.gz && cd .. &&
    aws s3 sync pretrained "$s3_prefix"/pretrained/ --delete --region "$REGION"

aws s3 cp "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/data/NEU-CLS.zip" "$s3_prefix"/data/ --source-region "$SOURCE_REGION"
aws s3 cp "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/data/NEU-DET.zip" "$s3_prefix"/data/ --source-region "$SOURCE_REGION"

echo "Add docs to build"
aws s3 sync "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/docs" "$s3_prefix"/docs --delete --region "$REGION"
aws s3 sync "s3://sagemaker-solutions-artifacts/sagemaker-defect-detection/docs" "$s3_prefix"/build/docs --delete --region "$REGION"
