import cv2
import os
import numpy as np

#path where the dataset is stored
path="../dataset/mnist_images/training/"
x=[]
y=[]

for i in range(10):
    path=f"../dataset/mnist_images/training/{i}/"
    for file in os.listdir(path):
        file_path=path+file
        
        img=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        
        #normalize
        img = img / 255.0

        # flatten to 784 vector
        img = img.flatten()
        x.append(img)
        y.append(i)
x=np.array(x)
y=np.array(y)
np.savez("mnist_dataset_train.npz", X=x, Y=y)
   