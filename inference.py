import os
import cv2
from argparse import ArgumentParser
import sys

sys.path.insert(0, "/home/user/client_projects/15secondsoffame/ai-features/fpage/ibug")
from ibug.face_detection import RetinaFacePredictor
from ibug.age_estimation.inference.age_estimator import AgeEstimator
import pandas as pd
from typing import Any, Dict
from tqdm import tqdm 
import json

def detect_faces_dataset(image_dir: str, selfies: pd.DataFrame, colname: str,
                        offset: int, limit: int, face_detector: Any, age_estimator: Any) -> Dict:
    result = {}
    for selfie in tqdm(selfies[colname][offset:offset+limit]):
        try:
            image = cv2.imread(os.path.join(image_dir, selfie))
            faces = face_detector(image, rgb=False)
            age, masks = age_estimator.predict_img(image, faces, rgb=False)
            result[selfie] = age[0].item()
        except Exception as e:
            print(str(e))
            print(os.path.join(image_dir, selfie))
            result[selfie] = -1
    return result 


if __name__ == "__main__":
    face_detector = RetinaFacePredictor(
        threshold=0.8,
        device="cuda:0",
        model=(RetinaFacePredictor.get_model("mobilenet0.25")),
    )
    age_estimator = AgeEstimator(
        device="cuda:0",
        ckpt=None,
        encoder="resnet50",
        decoder="fcn",
        age_classes=97,
        face_classes=14,
    )
    selfies = pd.read_csv("/home/user/client_projects/15secondsoffame/ai-features/ai-identification/test_data.csv")
    cur_dir = "/home/user/client_projects/15secondsoffame/ai-features/ai-identification/matches_data_"
    result_file = "/home/user/client_projects/15secondsoffame/ai-features/appearances_fpage.json"
    offset=0
    limit=10000
    detection_result = detect_faces_dataset(cur_dir, selfies, 'thumbnail_img', offset, limit, 
    face_detector, age_estimator)
    print(detection_result)
    with open(result_file, "w") as f:
        json.dump(detection_result, f)
