# DriverDrowsinessDetection <br/>

### This repository contains the implementation of a Driver Drowsiness Detection System using Neural Networks. The system is designed to monitor a driver's facial expressions and alert them in real-time if signs of drowsiness are detected. It utilizes deep learning techniques to analyze the driver's facial features and predict their level of alertness. <br/>

### The dataset used has been taken from kaggle <br/>
Link of the dataset: https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd <br/>
Properties of the dataset: <br/>
• RGB images <br/>
• 2 classes (Drowsy & Non Drowsy) <br/>
• Size of image : 227 x 227 <br/>
• More than 41,790 images in total <br/>

##  Our implementation consists of the following steps: <br/>
• Extracting facial landmarks using dlib  <br/>
• Calculating various ratios that are considered to be the estimators of the drowsiness level of an individual <br/>
   ### The various ratios used were: <br/>
    • Eye aspect ratio
    • Mouth aspect ratio 
    • Pupil circularity 
    • Mouth over Eye ratio 
• Training a neural network that takes as input the ratios obtained in the previous step and asigns a score between 0 and 1 denoting the drowsiness level ( 1 being drowsy ) <br/>
   ### The neural net consists of: <br/>
    • Layers : 9 hidden layers, 1 input layer, 1 output layer 
    • Activation Function : RELU  
    • Last Layer activation function : Sigmoid 
    • Loss Function : Binary Cross Entropy Loss  
    • Optimizer : Adam  
• Maintaining an Exponential Moving Average (EMA) of the drowsiness score obtained from the output of the neural network on the individual frames captured by the driver's camera sequentially and producing an alert message when the EMA crosses a certain threshold. <br/>
   ### EMA features: <br/>
    • Weightage : 0.6 for new frame 
    • EMA initialization value : 0.0 (Considering driver is not drowsy at the initial moment) 
    • EMA refresh rate : 10 frames 

### Frameworks used: Pytorch, Scikit-Learn, dlib, opencv, numpy, pandas, imutils <br/>
