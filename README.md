# Visual Inspection Automation with Amazon SageMaker

<p align="center">
  <a href="https://github.com/awslabs/sagemaker-defect-detection/actions"><img alt="Actions Status" src="https://github.com/awslabs/sagemaker-defect-detection/workflows/CI/badge.svg"></a>
  <a href="https://github.com/awslabs/sagemaker-defect-detection/actions"><img alt="CodeQL Status" src="https://github.com/awslabs/sagemaker-defect-detection/workflows/CodeQL/badge.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <br>
  <a href="https://github.com/awslabs/sagemaker-defect-detection/blob/mainline/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/github/license/awslabs/sagemaker-defect-detection.svg"></a>
  <a href="https://github.com/awslabs/sagemaker-defect-detection/graphs/commit-activity"><img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"></a>
  <a href="https://github.com/awslabs/sagemaker-defect-detection/issues/new?assignees=ehsanmok&labels=question&template=questions-or-general-feedbacks.md&title=%5BGeneral%5D"><img alt="AMA" src="https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg"></a>
 </p>

This solution detects product defects with an end-to-end Deep Learning workflow for quality control in manufacturing process. The solution takes input of product images and identifies defect regions with bounding boxes. In particular, this solution uses an implementation of  [An End-to-End Steel Surface Defect Detection](https://ieeexplore.ieee.org/document/8709818) on [NEU surface defect database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) (see [references](#references)) in [PyTorch](https://pytorch.org/) using [PyTorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).


## Overview

### How Does the Input Data Look Like?

Input is an image of a defective / non-defective product. Here is an example used in the demo

<p align="center">
  <img src="docs/sample2.png" alt="Solution Architecture" width="600" height="300">
</p>

### How to Prepare Your Data to Feed into the Model?

There is data preparation and preprocessing steps and should be followed in the notebook.

### What Are the Outputs?

Classifications and detections of defects and type of defects with bounding boxes (if there is any).

### What is the Estimated Cost?

Running the solution end-to-end costs around than $8 USD. Please make sure you have read the cleaning up part [here](#cleaning-up).

### How does the Data Flow Look Like?

<p align="center">
  <img src="docs/data_flow.png" alt="Solution Architecture" width="600" height="300">
</p>

# Solution Details

## Background

According to the [Gartner](https://www.gartner.com/smarterwithgartner/gartner-top-10-strategic-technology-trends-for-2020/),  hyper-automation is the number one trend in 2020 and will continue advancing in future. When it comes to manufacturing, one of the main barriers to hyper-automation is in areas where Human involvements is still struggling to be reduced and intelligent systems have hard times to become on-par with Human visual recognition abilities and become mainstream, despite great advancement of Deep Learning in Computer Vision. This is mainly due to lack of enough annotated data (or when data is sparse) in areas such as _Quality Control_ sections where trained Human eyes still dominates.


## What is Visual Inspection?

The **analysis of products on the production line for the purpose of Quality Control**. Visual inspection can also be used for internal and external assessment of the various equipment in a production facility such as storage tanks, pressure vessels, piping, and other equipment ([source](https://nanonets.com/blog/ai-visual-inspection/)) which expands to many industries from Electronics, Medical, Food and Raw Materials.

### What are the Problems?

Some of the major problems are

#### Human Visual Error

**Human visual inspection error** is a major factor in this area. According to this [report](https://www.osti.gov/servlets/purl/1476816)

> Most inspection tasks are much more complex and typically exhibit error rates of 20% to 30% (Drury & Fox, 1975)

which directly translates to *cost*.
#### Cost

According to some [estimate](https://www.glassdoor.co.in/Salaries/us-quality-control-inspector-salary-SRCH_IL.0,2_IN1_KO3,28.htm), a trained quality inspector salary varies between 26K (US) - 60K per year.

## Getting Started

You will need an AWS account to use this solution. Sign up for an account [here](https://aws.amazon.com/).

To run this JumpStart 1P Solution and have the infrastructure deploy to your AWS account you will need to create an active SageMaker Studio instance (see [Onboard to Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html)). When your Studio instance is *Ready*, use the instructions in [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) to 1-Click Launch the solution.

The solution artifacts are included in this GitHub repository for reference.

*Note*: Solutions are available in most regions including us-west-2, and us-east-1.

**Caution**: Cloning this GitHub repository and running the code manually could lead to unexpected issues! Use the AWS CloudFormation template. You'll get an Amazon SageMaker Notebook instance that's been correctly setup and configured to access the other resources in the solution.

## What Does this Solution Offer?

This solution offers an implementation of the state-of-the-art Deep Learning approach for automatic *Steel Surface Defect Detection* using **Amazon SageMaker**. The [model](https://ieeexplore.ieee.org/document/8709818) enhances [Faster RCNN](https://arxiv.org/abs/1506.01497) and output possible defects in an image of surface of a steel. The **NEU surface defect database**(see [references](#references)), is a *balanced* dataset which contains

> Six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The database includes 1,800 grayscale images: 300 samples each of six different kinds of typical surface defects

This solution trains a classifier on **NEU-CLS** dataset as well as a detector on **NEU-DET** dataset. Here is a sample images of the six classes

<p align="center">
  <img src="docs/data.png" alt="Sample data" width="500" height="500">
</p>
<p align="center">
  <a href="http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html"> NEU sample</a>
</p>

and here are the sample detection results

<p align="center">
  <img src="docs/sample3.png" alt="sample3" width="600" height="300"/>
</p>

## Contents

* `cloudformation/`
    * `defect-detection.yaml`: The root cloudformation nested stack which creates the AWS stack for this solution
    * `defect-detection-sagemaker-notebook-instance.yaml`: Creates SageMaker notebook instance
    * `defect-detection-permissions.yaml`: Manages all the permission necessary to launch the stack
    * `defect-detection-endpoint.yaml`: Creates demo endpoint using in `0_demo.ipynb`
    * `solution-assistant`: Deletes the created resources such as endpoint, S3 bucket etc. during cleanup
* `src/`
    * `prepare_data/`: Data prepartion for NEU datasets
    * `sagemaker_defect_detection/`: Main package
        * `dataset`: Contains NEU dataset handling
        * `models`: Contains the MFN model code
        * `utils`: Various utilities for visualization and coco evaluation
        * `classifier.py`: For classification task
        * `detector.py`: For detection task
        * `transforms.py`: Contains the image transformations used in training
* `notebooks/`: All the notebooks described above
* `scripts/`: Various scripts for training and building

## Architecture Overview

Here is architecture for the end-to-end training and deployment process

<p align="center">
  <img src="docs/train_arch.png" alt="Solution Architecture" width="800" height="400">
</p>

1. The input data located in an [Amazon S3](https://aws.amazon.com/s3/) bucket
2. The provided [SageMaker notebook](source/deep_demand_forecast.ipynb) that gets the input data and launches the later stages below
3. **Training Classifier and Detector models** and evaluating its results using Amazon SageMaker. If desired, one can deploy the trained models and create SageMaker endpoints
4. **SageMaker endpoint** created from the previous step is an [HTTPS endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html) and is capable of producing predictions
5.  Monitoring the training and deployed model via [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)

## Cleaning up

When you've finished with this solution, make sure that you delete all unwanted AWS resources. AWS CloudFormation can be used to automatically delete all standard resources that have been created by the solution and notebook. Go to the AWS CloudFormation Console, and delete the parent stack. Choosing to delete the parent stack will automatically delete the nested stacks.

**Caution:** You need to manually delete any extra resources that you may have created in this notebook. Some examples include, extra Amazon S3 buckets (to the solution's default bucket), extra Amazon SageMaker endpoints (using a custom name).

## Customization

For using your own data, make sure it is labeled and is a *relatively* balanced dataset.

## Useful Links

* [Amazon SageMaker Getting Started](https://aws.amazon.com/sagemaker/getting-started/)
* [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
* [Amazon SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/en/stable/)
* [AWS CloudFormation User Guide](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html)

### References

* K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science, vol. 285, pp. 858-864, Nov. 2013.

* Yu He, Kechen Song, Qinggang Meng, Yunhui Yan, “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,” IEEE Transactions on Instrumentation and Measuremente, 2020,69(4),1493-1504.

* Hongwen Dong, Kechen Song, Yu He, Jing Xu, Yunhui Yan, Qinggang Meng, “PGA-Net: Pyramid Feature Fusion and Global Context Attention Network for Automated Surface Defect Detection,” IEEE Transactions on Industrial Informatics, 2020.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
