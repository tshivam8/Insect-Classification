# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:50:57 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:05:24 2018

@author: Administrator
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import glob

green=(0,255,0)
red=(255,0,0)
blue=(0,0,255)


def find_biggest_contour(image):
    image=image.copy()
    
    contours , hierarchy=cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes=[(cv2.contourArea(contour),contour) for contour in contours]
    biggest_contour=max(contour_sizes,key=lambda x:x[0])[1]
    mask=np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[biggest_contour],-1,(0,255,0),-1)
    return biggest_contour,mask

def overlay_mask(mask,image):
	rgb_mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
	img=cv2.addWeighted(rgb_mask,0.5,image,0.5,0)
	return img

def rectangle_contour(image, cnt):
    image_with_rectangle = image.copy()

    x, y, w, h = cv2.boundingRect(cnt)

    img = cv2.rectangle(image_with_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return img

def rot_rectangle_contour(image,cnt):
    image_with_rectangle = image.copy()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(image_with_rectangle,[box],0,(0,0,255),2)
    return img

def show(image):

	plt.figure(figsize=(10,10))
	plt.imshow(image,interpolation='nearest')
    
def draw_redshade(image):

	#PRE PROCESSING OF IMAGE

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=1000/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	min_color=np.array([100,0,80])
	max_color=np.array([255,10,255])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2=np.array([100,170,80])
	max_color2=np.array([255,190,255])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

	overlay=overlay_mask(mask_cleaned,image)

	circled=rectangle_contour(overlay,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

	return bgr




def draw_greenshade(image):

	#PRE PROCESSING OF IMAGE
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    maxsize=max(image.shape)
    scale= 500/maxsize

    image=cv2.resize(image,None,fx=scale,fy=scale)

    image_blur=cv2.GaussianBlur(image,(7,7),0)

    image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

    min_color=np.array([15,20,65])
    max_color=np.array([25,55,245])
    
    mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

    min_color2=np.array([55,50,100])
    max_color2=np.array([255,255,255])

    mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

    mask=mask1+mask2

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

    big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

    overlay=overlay_mask(mask_cleaned,image)

    circled=rot_rectangle_contour(overlay,big_contour)

    show(circled)

    bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)
    return bgr




#input image
from glob import glob

#img_mask = 'C:/Users/Administrator/Desktop/Detection/Rice/Input/3 Anomala dimidiata Hope/*.jpg'
img_mask = r"Datasets\Wang 9 classes_Augmented\Auchenorrhyncha\*.jpg"

img_names = glob(img_mask)    
#print(img_names)
#insect = cv2.imread(img_mask)
#result_insect=draw_redshade(insect)
#cv2.imwrite('C:/Users/Administrator/Desktop/Detection/Outputs/2/1.jpg',result_insect)
i=1
for file in img_names:
    insect = cv2.imread(file)
    result_insect= draw_redshade(insect)
    #cv2.imwrite('C:/Users/Administrator/Desktop/Detection/Rice/Output/new/New_{:>01}.jpg'.format(i),result_insect)
    if not os.path.exists('new'):
        os.mkdir('new')
    cv2.imwrite(r'new\New_{:>01}.jpg'.format(i),result_insect)
    i=i+1
  

  

