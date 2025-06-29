import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils    
import math


def EAR(eye):
    A = dist.euclidean([eye[2], eye[3]], [eye[10], eye[11]])
    B = dist.euclidean([eye[4], eye[5]], [eye[8], eye[9]])
    C = dist.euclidean([eye[0], eye[1]], [eye[6], eye[7]])
    ear = (A+B)/(2*C)
    return ear

def MAR(mouth):
    A1=dist.euclidean([mouth[28], mouth[29]], [mouth[36], mouth[37]])
    A2=dist.euclidean([mouth[26], mouth[27]], [mouth[38], mouth[39]])
    A3=dist.euclidean([mouth[30], mouth[31]], [mouth[34], mouth[35]])
    A=(A1+A2+A3)/3.0
    B1=dist.euclidean([mouth[2], mouth[3]],[mouth[10], mouth[11]])
    B2=dist.euclidean([mouth[22], mouth[23]],[mouth[14], mouth[15]])
    B3=dist.euclidean([mouth[0], mouth[1]],[mouth[12], mouth[13]]) 
    B=(B1+B2+B3)/3.0 
    res= A/B
    return res

def MOE(Ear, Mar):
    return Mar / Ear

def PUC(eye):
    A = dist.euclidean([eye[2], eye[3]], [eye[8], eye[9]])
    B = dist.euclidean([eye[4], eye[5]], [eye[10], eye[11]])
    radius = (A + B) / 4.0
    perimeter = dist.euclidean([eye[0], eye[1]], [eye[2], eye[3]]) + dist.euclidean([eye[2], eye[3]], [eye[4], eye[5]]) + dist.euclidean([eye[4], eye[5]], [eye[6], eye[7]]) + dist.euclidean([eye[6], eye[7]], [eye[8], eye[9]]) + dist.euclidean([eye[8], eye[9]], [eye[10], eye[11]]) + dist.euclidean([eye[10], eye[11]], [eye[0], eye[1]])
    area = math.pi * (radius ** 2)
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity

ptp= "68 face landmarks.dat"
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(M_start,M_end)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']


df = pd.read_csv('dataset_of_coordinates.csv',header=None)


df_left_eye = df.iloc[: ,2*L_start:2*L_end]
df_right_eye = df.iloc[: ,2*R_start:2*R_end]
df_mouth = df.iloc[: ,2*M_start:2*M_end]
Y = df.iloc[: , -1:]

df_left_eye.columns = range(0,df_left_eye.shape[1])
df_right_eye.columns = range(0,df_right_eye.shape[1])
df_mouth.columns = range(0,df_mouth.shape[1])


def make_list(x):
    l = []
    for i in range(0 , x + 1):
        l.append(i)
    return l


df_left_eye['EAR'] = df_left_eye.apply(lambda row : EAR(row[make_list(11)].values), axis=1)
df_left_eye['PUC'] = df_left_eye.apply(lambda row : PUC(row[make_list(11)].values), axis=1)
df_right_eye['EAR'] = df_right_eye.apply(lambda row: EAR(row[make_list(11)].values), axis=1)
df_right_eye['PUC'] = df_right_eye.apply(lambda row: PUC(row[make_list(11)].values), axis=1)
df_mouth["MAR"]=df_mouth.apply(lambda row: MAR(row[make_list(39)].values), axis=1)


df_final = pd.DataFrame().assign(lEAR=df_left_eye['EAR'], rEAR=df_right_eye['EAR'] , MAR = df_mouth['MAR'])


df_final["MOE"] = 2 * df_final["MAR"] / (df_final['lEAR'] + df_final['rEAR'])
df_final["l_PUC"] = df_left_eye["PUC"]
df_final["r_PUC"] = df_right_eye["PUC"]
df_final['label']=Y


df_final.to_csv('dataset_parameters.csv', header=True, index=False)