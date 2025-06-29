


from os import listdir
import cv2
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils    
import math
import dlib



ptp= "68 face landmarks.dat" 
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(ptp) 





dr_landmarks=[]
path = "C:\\Users\\91993\\Summer_Analytics\\CNA_proj\\vid\\(DDD)\\Drowsy\\"
for images in listdir(path):
    filename = path + images
    img = cv2.imread(filename= filename)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    face = face_detector(img)
    if(len(face)==0):
        continue
    face = face[0]
    landmarks = landmark_detector(img , face)
    landmarks = face_utils.shape_to_np(landmarks)
    landmarks = landmarks.tolist()
    dr_landmarks.append(landmarks)
    




list_drowsy = []
for landmarks in dr_landmarks:
    coordinates = []
    for landmark in landmarks:
        coordinates.append(landmark[0])
        coordinates.append(landmark[1])
    list_drowsy.append(coordinates)





for data in list_drowsy:
    data.append(1)





ndr_landmarks=[]
path = "C:\\Users\\91993\\Summer_Analytics\\CNA_proj\\vid\\(DDD)\\Non_Drowsy\\"
for images in listdir(path):
    filename = path + images
    img = cv2.imread(filename= filename)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    face = face_detector(img)
    if(len(face)==0):
        continue
    face = face[0]
    landmarks = landmark_detector(img , face)
    landmarks = face_utils.shape_to_np(landmarks)
    landmarks = landmarks.tolist()
    ndr_landmarks.append(landmarks)
    





list_non_drowsy = []
for landmarks in ndr_landmarks:
    coordinates = []
    for landmark in landmarks:
        coordinates.append(landmark[0])
        coordinates.append(landmark[1])
    list_non_drowsy.append(coordinates)





for data in list_non_drowsy:
    data.append(0)





dataset_of_coordinates = list_drowsy + list_non_drowsy





dataset_of_coordinates = np.array(dataset_of_coordinates)
np.savetxt("dataset_of_coordinates.csv", dataset_of_coordinates, delimiter = ", ", fmt = "%i")

