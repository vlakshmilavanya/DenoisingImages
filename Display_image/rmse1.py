# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:33:15 2020

@author: reeth
"""
import os
import numpy as np
import cv2
import pylab as p
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import matplotlib


#from pytesseract import image_to_string
#from io import BytesIO
#import base64
#from uuid import uuid4

#from flask import Flask, request, render_template, send_from_directory
#APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#target = os.path.join(APP_ROOT, 'noisedImages/')
#target1 = os.path.join(APP_ROOT, 'DenoisedImages/')
#for i in  os.path.dir(target):
    #print(i)
#print(t)
img = cv2.imread(os.path.join("noisedImages","1.png"))
def load_images_from_folder(folder,filename):
    images = []
    img = cv2.imread(os.path.join(folder,filename))
        
def clean_image(input_img):
    plt.figure(figsize=(15, 30))
    plt.subplot(121)
    plt.imshow(input_img, cmap = cm.gray)
   # print(img_final.shape)

    kernel = np.ones((4,4), np.uint8) 

    #erode will remove only background
    

    img_erode  = 255 - cv2.erode(255 - input_img, kernel,iterations = 1)

    img_sub = cv2.add(input_img, - img_erode)

    #need to choose threshold automatically?

    _, img_thresh = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)

    mask = img_thresh == 0                                     

    img_final = np.where(mask, input_img, 255)
    #filte=cv2.Canny(img_final,100,200)
    plt.figure(figsize=(15, 30))
    plt.subplot(121)
    plt.imshow(img_final, cmap = cm.gray)
    print(img_final.shape)
    #_, img_thresh1 = cv2.threshold(filte, 255, 255, cv2.THRESH_BINARY)
    #mask = img_thresh1 == 0                                     
    #img_final1 = np.where(mask, img_final, 255)
    #plt.subplot(122)
    #plt.imshow(img_final1, cmap = cm.gray)
    #plt.show() 
    return img_final
def rmse1(true_images, pred_images):
    result = n = 0
    result += np.sum(true_images.ravel()/255.0 - pred_images.ravel()/255.0**2)
    n += len(true_images.ravel())

    return (result / float(n))**0.5
print(rmse1(img,clean_image(img)))
print(img)