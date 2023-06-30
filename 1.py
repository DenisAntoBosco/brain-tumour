import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
file_list = []
class_list = []

DATADIR = "data"

# All the categories you want your neural network to detect
CATEGORIES = ["glioma","meningioma","notumor","pituitary"]

# The size of the images that your neural network will use
IMG_SIZE = 50
def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges
# Checking or all images in the data folder


import cv2
a=input("Enter image path")
img_array = cv2.imread(a,0)
img_array = cv2.resize(img_array, (250, 250))
print("ORIGINAL IMAGE")
cv2.imshow("",img_array)
cv2.waitKey(0)

#img_array = cv2.Canny(img_array, threshold1=50, threshold2=10)

clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
img_array = clahe.apply(img_array)
median  = cv2.medianBlur(img_array.astype('uint8'), 5)
median = 255-median
ret,thresh = cv2.threshold(median.astype('uint8'),165,255,cv2.THRESH_BINARY_INV)
img_array=cv2.fastNlMeansDenoising(img_array)
print(img_array)
print("PRE-PROCESSED IMAGE")
cv2.imshow("",img_array)
cv2.waitKey(0)



training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
                img_array = clahe.apply(img_array)
                median  = cv2.medianBlur(img_array.astype('uint8'), 5)
                median = 255-median
                ret,thresh = cv2.threshold(median.astype('uint8'),165,255,cv2.THRESH_BINARY_INV)
                #img_array=cv2.equalizeHist(img_array)
                #img_array = cv2.Canny(img_array, threshold1=30, threshold2=40)
                #img_array = cv2.medianBlur(img_array,1)
                new_array = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
#X = np.array(X)
# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


