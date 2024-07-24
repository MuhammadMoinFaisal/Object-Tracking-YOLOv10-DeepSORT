#Import All the Required Libraries
import cv2
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

class ObjectTracking:
    def __init__(self):
        self.className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                          "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                          "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
                          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                          "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.colors = np.random.randint(0,255, size=(len(self.className),3))

    def initialize_deepsort(self):
        #Create the DeepSORT configuration object and load settings from the YAML file
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        #Initialize the DeepSORT tracker
        deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist = cfg_deep.DEEPSORT.MAX_DIST,
                            #minimum confidence parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                            min_confidence = cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap = cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance = cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            #max_age: if an object tracing ID is lost, this parameter determines how many frames the tracker should wait before assigning a new iD
                            max_age = cfg_deep.DEEPSORT.MAX_AGE,
                            n_init = cfg_deep.DEEPSORT.N_INIT,
                            #nn_budget: It sets the budget for nearest-neighbor search
                            nn_budget = cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda = False
                            )
        return deepsort
    def draw_boxes(self, frame, bbox_xyxy, identities = None, classID = None, offset = (0,0)):
        height, weight, _ = frame.shape
        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            y1 += offset[0]
            x2 += offset[0]
            y2 += offset[0]
            #Find the center coordinates of the bounding boxes
            cx, cy = int((x1+x2)/2), int((y1 + y2)/2)
            cv2.circle(frame, (cx, cy), 2, (0,255,0), cv2.FILLED)
            clsID = int(classID[i]) if classID is not None else 0
            id = int(identities[i]) if identities is not None else 0
            #color = self.color
            color = self.colors[clsID]
            B, G, R = map(int, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            className = self.className
            name = className[clsID]
            label = str(id) + ":" + name
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1,y1), c2, (B, G, R), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        return frame