

This repository contains the implementation of the Novel Proposed CNN for detecting grasp poses : 


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
$ cd CAPS-FRAMEWORK- 
$ pip install -r requirements.txt
```


## Datasets

This repository uses the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) for grasp pose estimation algorithms
and uses a synthetic dataset for object detection, the dataset is generated using this [pipeline] (https://github.com/KulunuOS/gazebo_dataset_generation)and our CAD data.  


#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`


## Grasping Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

##Grasping Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```


## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python run_grasp_generator.py
```
```bash
python run_realtime.py --network "<ADD_PATH_TO>/epoch_08_iou_1.00"

```

```bash
$ cd robotic-grasping-CNN
$ python run_realtime.py --network "<ADD_PATH_TO_TRAINED-MODEL>/epoch_08_iou_1.00"
```


