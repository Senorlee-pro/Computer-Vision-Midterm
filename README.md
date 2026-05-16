# Computer-Vision-Midterm
## Environment Setup
You may want to create a new `conda` environment for this project.

```bash
conda create -n myenv python=3.10
conda activate myenv
```

Then, enter the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```
## Dataset
This project requires two datasets, **Oxford-iiit-pet** and **VisDrone**. You should arrange the dataset structure as follows:

```bash
root/
├── archive/
│   ├── VisDrone2019-DET-test-dev
│   ├── VisDrone2019-DET-train
│   └── VisDrone2019-DET-val
├── oxford-iiit-pet
├── task1_main.py
└── ...
```

## Project Structure
- **`task1_main.py`** - Full implementation of task 1.
- **`task2_main.py`** - Implementation of task 2, especially the part of finetuning **YOLO**.
- **`task2_track.py`** - Implementation of the tracking part in task 2.
- **`task3_loss.py`** - Implementation of the loss functions in task 3, including Dice loss and Combined loss.
- **`task3_main.py`** - Implementation of the main process of training U-Net in task 3.
- **`task3_unet.py`** - Definition of the U-Net model structure in task 3.

## Get started
### Task 1
You only need to run `task1_main.py` to train **ResNet** and **ViT-Tiny**.

Note that all the required variables and hyperparameters are listed at the very beginning of the code entrance, with their names fully capitalized.

Specifically, you can switch the value of `MODEL` between `"resnet"` and `"vit"` to train either **ResNet-18** or **ViT-Tiny**.

### Task 2
For finetuning **YOLOv8**, run `task2_main.py`. Note that `PATIENCE` refers to the tolerance interval regarding early stopping mechanism.

The training outcome will automatically stored in `runs/detect/visdrone_yolov8`, you can find the corresponding weights in the subdirectory `weights`.

For the subsequent tracking, run `task2_track.py`. You should set `VIDEO_PATH`, `MODEL_PATH`, `OUTPUT_VIDEO` and `LINE_Y` correctly for function `run_tracking`, it will output a video with each frame annotated by bounding boxes, and create a horizontal line at `LINE_Y` to count the objects crossing the line as well.
### Task 3
You only need to run `task3_main.py` to train U-Net from scratch.

Note that the program will train U-Net with all of the three loss functions by default. You can reset the loss function configuration in `loss_configs`.

**All programs will automatically evaluate on the validation and test sets, and output the metrics.**
