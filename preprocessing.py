import os
from ibug.face_detection import RetinaFacePredictor
import pandas as pd
import torch
from tqdm import tqdm
import cv2

#####################################################################
# RetinaFacePredictor returns:
# left, top, right, bottom, detection confidence 
# 5 landmarks(x1, y1, x2, y2, ..., x5, y5)
####################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def add_bbox_column_to_dataset(dataset: str, data_dir: str, device):
    face_detector =  RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=(RetinaFacePredictor.get_model("mobilenet0.25")),
    )
    data = pd.read_csv(dataset)
    data['bbox'] = ','.join('0'*15)
    for i in range(data.shape[0]):
        img = cv2.imread(os.path.join(data_dir, data.loc[i, 'image']))
        try:
            bboxes = face_detector(img, rgb=False)
            data.loc[i, 'bbox'] = ','.join(map(str, bboxes[0]))
        except:
            print(bboxes)
    data.to_csv(dataset, index=False)
        

if __name__ == "__main__":
    dataset = "/home/user/client_projects/15secondsoffame/ai-features/age_data_train.csv"
    data_dir = "/home/user/client_projects/15secondsoffame/ai-features/selfie"
    add_bbox_column_to_dataset(dataset, data_dir, device)


       