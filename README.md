<H1 align="center">
Object Tracking using YOLOv10 & DeepSORT </H1>

## Installation
1. Clone the YOLOv10 github repo.
```
https://github.com/THU-MIG/yolov10.git
```
2. Set YOLOv10 folder as current directory
```
cd yolov10
```
3. Install all the required packages
```
pip install -r requirements.txt
```
4. Perform Object Detection using YOLOv10
```
python object_detection.py
```
5. Download DeepSORT files from the Google Drive and add them into 
```
https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
```
- After downloading the DeepSORT Zip file from the drive, unzip it go into the subfolders and place the deep_sort_pytorch folder into the yolov10 folder

6. Install additional libraries to implement object tracking using DeepSORT
```
pip install easydict
```
```
pip install "numpy<1.24"
```
7. Perform object tracking using YOLOv10 and DeepSORT algorithm.
```
python objectTracking.py
```

