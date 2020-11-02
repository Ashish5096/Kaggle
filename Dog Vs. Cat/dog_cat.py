import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle



DATADIR ="/home/ashish/SE_Project/kagglecatsanddogs"

CATEGORIES=["Dog","Cat"]
img_size=100
training_data=[]

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category) 
        class_num=CATEGORIES.index(category)
       
        for image in os.listdir(path):
            try:
                img= cv2.imread(os.path.join(path,image),0)
                new_img= cv2.resize(img,(img_size,img_size))
                training_data.append([new_img,class_num])
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)

X=[]
Y=[]

for features,label in training_data:
    X.append(features)
    Y.append(label)


X=np.array(X).reshape(-1, img_size,img_size, 1)

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

