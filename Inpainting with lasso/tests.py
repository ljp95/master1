
""" Created by ljp95 and Elodie Souksava on 05.26.2019 """

from inpainting_lasso import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb

import time
from sklearn import linear_model

################### Comparing Regularization with dfferent norms ##############
# choice of classs
class1 = 1
class2 = 7
# extract data with chosen classs
datax,datay = load_usps('USPS_train.txt')
trainx, trainy = extract_data(class1,class2,datax,datay)
datax,datay = load_usps('USPS_test.txt')
testx,testy = extract_data(class1,class2,datax,datay)

##### Logistic regression
reg = linear_model.LogisticRegression()
reg.fit(trainx,trainy)
name = 'logistic regression {} vs {}'.format(class1,class2)
infos_usp(name,reg,reg.coef_[0],trainx,trainy,testx,testy)


##### Ridge regression
alpha = 10
ridge = linear_model.Ridge(alpha = alpha)
ridge.fit(trainx,trainy)
name = 'regression ridge {} vs {}'.format(class1,class2)
infos_usp(name,ridge,ridge.coef_,trainx,trainy,testx,testy)
print("alpha = {}".format(alpha))


##### Lasso
alpha = 0.001
lasso = linear_model.Lasso(alpha = alpha)
lasso.fit(trainx,trainy)
name = 'lasso {} vs {}'.format(class1,class2)
infos_usp(name,lasso,lasso.coef_,trainx,trainy,testx,testy)
print("alpha = {}".format(alpha))


########################### LASSO AND INPAINTING ###############################

h = 10

# choose image
file = 'ocean.jpg'
image,mean,std = read_image(file)
plt.figure();show_image(image,mean,std);plt.title("original image")


# building all patches
vectors,complete_vectors,incomplete_vectors = construct_vectors(image,h)
trainx = complete_vectors

##### Test on a patch
# choosing a point
line,col = 205,205 # horizon
line,col = 100,395 # cloud
line,col = 285,285 # ocean

# original patch
patch = get_patch(line,col,h,image)
plt.figure();show_image(patch,mean,std);plt.title("original patch")
vector = patch_to_vector(patch)

# approximating the original patch
alpha = 0.00001
max_iter = 1000
w = approximation(vector,trainx,h,mean,std,alpha)
infos_inpainting(w,alpha)


# noisy patch 
patch_noisy = noise(patch,0.5)
vector_noisy = patch_to_vector(patch_noisy)
plt.figure();show_image(patch_noisy,mean,std);plt.title("noisy patch")

# approximation on noisy patch
w = approximation(vector_noisy,trainx,h,mean,std,alpha=alpha,noise = 1)
infos_inpainting(w,alpha)


##### Test on the image
### choose between adding noise randomly in the image or a full rectangle
# adding noise in the image
prc = 0.5
image_noisy = noise(image,prc)
show_image(image_noisy,mean,std)

# adding a noisy rectangle
height = 20
length = 20
image_noisy = delete_rect(image,line,col,height,length)
show_image(image_noisy,mean,std)
zoom(image_noisy,mean,std,line,col,height*2,length*2)


# simple filling
begin = time.time()
vectors,trainx,incomplete_patches = construct_vectors(image,h)
image2 = simple_filling(image_noisy,mean,std,trainx,line,col,height,length,h) 
print("Approximation in {}secs".format(time.time()-begin))  
show_image(image2,mean,std) 

# better filling
image_noisy_rgb = plt.imread(file)*1.0
begin = time.time()
vectors,trainx,incomplete_patches = construct_vectors(image,h)
image2 = better_filling(image_noisy_rgb,image_noisy,mean,std,trainx,line,col,height,length,h) 
print("Approximation in {}secs".format(time.time()-begin))  
show_image(image2,mean,std) 
        
        
                
                