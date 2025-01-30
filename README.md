## Description 

CAPS Framwork is a framework that allows for interchangable grasping between Humans and Robots in the context of Human robot collaboration assembly process.

Our framework utilizes synthetic data generation, object detection and grasp pose estimation, where all synchronously work to achieve dynamic interchangability between Human and Robots in assembly process as well as flexibilit.  


## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow


## Installation

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd CAPS-Framework
$ pip install -r requirements.txt
```


## Datasets

This repository uses the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) for grasp pose estimation algorithms
and uses a synthetic dataset for object detection, the dataset is generated using this [pipeline](https://github.com/KulunuOS/gazebo_dataset_generation) and our CAD data.  


#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`


## Grasping Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

## Grasping Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```


## Run Tasks

This repo has two main tasks, the object detection task which uses the YOLOV8 trained on our synthetic dataset for object deteection and a grasp pose estimation algorithm which is trained on the cornell grasping dataset. 


### Running Object Detection 
```bash
cd CAPS-Framework
```
The object detection code uses the ROS topics from our cameras so make sure to change the topic name.

```bash
python object_detection.py
```

### Running Grasping Pretrained Model 
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:


```bash
$ cd CAPS-Framework 
$ python run_realtime.py --network "<ADD_PATH_TO_TRAINED-MODEL>/epoch_08_iou_1.00"
```


