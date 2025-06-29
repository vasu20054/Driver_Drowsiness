get_ipython().run_cell_magic('javascript', '', '\nwindow.scroll_flag = true\nwindow.scroll_exit = false\nwindow.scroll_delay = 100\n\n$(".output_scroll").each(function() {\n    $(this)[0].scrollTop = $(this)[0].scrollHeight;\n});\n\nfunction callScrollToBottom() {\n    setTimeout(scrollToBottom, window.scroll_delay);\n}\n\nfunction scrollToBottom() {\n    if (window.scroll_exit) {\n        return;\n    }\n    if (!window.scroll_flag) {\n        callScrollToBottom();\n        return;\n    };\n    \n    $(".output_scroll").each(function() {\n        if (!$(this).attr(\'scroll_checkbox\')){\n            window.scroll_flag = true;\n            $(this).attr(\'scroll_checkbox\',true);\n            var div = document.createElement(\'div\');\n            var checkbox = document.createElement(\'input\');\n            checkbox.type = "checkbox";\n            checkbox.onclick = function(){window.scroll_flag = checkbox.checked}\n            checkbox.checked = "checked"\n            div.append("Auto-Scroll-To-Bottom: ");\n            div.append(checkbox);\n            $(this).parent().before(div);\n        }\n        \n        $(this)[0].scrollTop = $(this)[0].scrollHeight;\n    });\n    callScrollToBottom();\n}\nscrollToBottom();\n')


import pandas as pd
import numpy as np
import math
import cv2
import dlib
from imutils import face_utils
import imutils
from scipy.spatial import distance as dist
import winsound
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Para = pd.read_csv('Parameters.csv', header = None)
mean = Para[0].to_numpy()
std = Para[1].to_numpy()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.sigmoid(self.fc10(x))
        return x


net = Net()
net.load_state_dict(torch.load('model_weights_1_6.pth'))


def EAR(eye):
    A = abs(dist.euclidean(eye[1], eye[5]))
    B = abs(dist.euclidean(eye[2], eye[4]))
    C = abs(dist.euclidean(eye[0], eye[3]))
    ear = (A+B)/(2*C)
    return ear

def MAR(mouth):
    A1=dist.euclidean(mouth[14],mouth[18])
    A2=dist.euclidean(mouth[13],mouth[19])
    A3=dist.euclidean(mouth[15],mouth[17])
    A=(A1+A2+A3)/3.0
    
    B1=dist.euclidean(mouth[1],mouth[5])
    B2=dist.euclidean(mouth[11],mouth[7])
    B3=dist.euclidean(mouth[0],mouth[6]) 
    B=(B1+B2+B3)/3.0
    
    res= A/B
    return res
def MOE(EAR, MAR):
    return MAR / EAR
def PUC(eye):
    A = dist.euclidean(eye[1], eye[4])
    B = dist.euclidean(eye[2], eye[5])
    radius = (A + B) / 4.0
    perimeter = dist.euclidean(eye[0], eye[1]) + dist.euclidean(eye[1], eye[2]) + dist.euclidean(eye[2], eye[3]) + dist.euclidean(eye[3], eye[4]) + dist.euclidean(eye[4], eye[5]) + dist.euclidean(eye[5], eye[0])
    area = math.pi * (radius ** 2)
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity


ptp= "68 face landmarks.dat" 
face_detector = dlib.get_frontal_face_detector() 
landmark_detector = dlib.shape_predictor(ptp) 
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(M_start,M_end)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']


temp_initialise = 0.0
alert_threshold = 0.65
new_frame_weightage = 0.6
temp_refresh_rate = 10
mar_threshold = 0.35


take_frame = False
cam = cv2.VideoCapture(0)
temp = temp_initialise
frames_not_drowsy = 0
while 1:
    if (frames_not_drowsy == temp_refresh_rate):
        frames_not_drowsy = 0
        temp = 0
    #checking the frame , if it reaches the end , start again
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT) :
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    _ , frame = cam.read()
    frame = cv2.resize(frame , (720 , 640))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) != 1:
        continue
    for face in faces :
        shape = landmark_detector(gray, face) #applies landmark on all faces avialbe <forms objects>
        shape = face_utils.shape_to_np(shape) # convert all objects created into np.array
        print(type(shape[0]))
        #i.e. it forms corrdinates for 68 landmarks (objects to np.array)
        leftEye = shape[L_start:L_end]
        rightEye = shape[R_start:R_end]
        mouth=shape[M_start:M_end]
        leftEAR = EAR(leftEye)
        rightEAR = EAR(rightEye)
        avg_EAR = (leftEAR + rightEAR)/2.0
        left_circ = PUC(leftEye)
        right_circ = PUC(rightEye)
        Mar=MAR(mouth)
        moe = MOE(avg_EAR, Mar)
        if Mar < mar_threshold :
            Mar = 0
        else :
            Mar = 1
        arr = np.array([leftEAR, rightEAR, Mar, moe, left_circ,right_circ])
        arr = arr-mean
        arr = arr/std
        arr = np.reshape(arr, (1, -1))
        arr = torch.tensor(arr, dtype=torch.float32)
        net.eval()  # set the model to evaluation mode
        with torch.no_grad():
            output = net(arr)
            temp = temp * (1 - new_frame_weightage) + output.item() * new_frame_weightage
            print(temp)
            if (temp < alert_threshold):
                frames_not_drowsy += 1
                take_frame = False
            else:
                frames_not_drowsy = 0
                if (take_frame == True):
                    print("ALERT!!!")
                    winsound.Beep(440, 500)
                else:
                    take_frame = True
    cv2.imshow("Video" ,  gray)
    if cv2.waitKey(3) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows




