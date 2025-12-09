from roboflow import Roboflow
import os
rf = Roboflow(api_key="7ldwT6oeFy8rCd3QCntI")
project = rf.workspace("cath-oxqmx").project("vehicle-detection-ckrxi-aqcsy")
project.version(1).download("yolov8", location="dataset")