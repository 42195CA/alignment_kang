#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:37:09 2020

@author: liuxia
"""

import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
#from scipy.spatial.transform import Rotation as R

mainpath='/home/liuxia/Documents/kang/indoor'
ipath=mainpath+'/images/pano/image'
ppath=mainpath+'/scans/scan'


def get_M(r,t):
    theta=r[0]
    phi=r[1]
    gamma=r[2]            
    # extrinsic Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])
        
    RY = np.array([ [np.cos(phi), 0, np.sin(phi), 0],
                     [0, 1, 0, 0],
                     [-np.sin(phi), 0, np.cos(phi), 0],
                     [0, 0, 0, 1]])
        
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                     [np.sin(gamma), np.cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    Rot = np.dot(np.dot(RZ, RY),RX)
    T = np.array([[1, 0, 0, t[0]],
                 [0, 1, 0, t[1]],
                 [0, 0, 1, t[2]],
                 [0, 0, 0, 1]])

    return np.dot(Rot, T)
    
def xyz2pixel(data,w,h):
    r=np.sqrt(data[0,:]*data[0,:]+data[1,:]*data[1,:]+data[2,:]*data[2,:])
    th=np.arctan2(data[1,:],data[0,:])
    phi=np.arccos(data[2,:]/r)
    u = -1/(2*np.pi)*th+0.5;
    v = -1/np.pi*phi+0.5;
    return np.array([u,v,data[3,:]])


def getimageedgescore_sobelOperator(img):
    sobel = np.copy(img)
    size = sobel.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            for k in range(size[2]):                
                gx = (img[i - 1][j - 1][k] + 2*img[i][j - 1][k] + img[i + 1][j - 1][k]) - (img[i - 1][j + 1][k] + 2*img[i][j + 1][k] + img[i + 1][j + 1][k])
                gy = (img[i - 1][j - 1][k] + 2*img[i - 1][j][k] + img[i - 1][j + 1][k]) - (img[i + 1][j - 1][k] + 2*img[i + 1][j][k] + img[i + 1][j + 1][k])
                sobel[i][j][k]= min(255, np.sqrt(gx**2 + gy**2))
    return sobel

def getpointedgescore(P):
    return 1

def getcost():
    return 0

def getgradient():
    return 1

def getstepsize():
    return 1


def l2_imageixels(a,b,h,w):
    dx=np.abs(a[0]-b[0])
    dy=np.abs(a[1]-b[1])
    dx_n=h-dx;
    dy_n=w-dy;
    d1=np.sqrt(dx*dx+dy*dy)
    d2=np.sqrt(dx_n*dx_n+dy*dy)
    d3=np.sqrt(dx*dx+dy_n*dy_n)
    d4=np.sqrt(dx_n*dx_n+dy_n*dy_n)
    return np.amin([d1,d2,d3,d4])


# gaussian kernel
def kfunc(d,sigma):
    return np.exp(-0.5*d*d/sigma/sigma)/np.sqrt(2*np.pi)/sigma

# gaussian filter
def gfunc(x,y,sigma):
    return (np.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2,j-size[1]//2, sigma )
    return out/np.sum(out)

def getimage_gaussianFilter(image, filter):
    iw,ih,sid = image.shape    
    fw,fh = filter.shape    
    out = np.zeros((iw-fw+1,ih-fh+1,sid))
    for d in range(sid):
        for w in range(ih-fh+1):
            for h in range(iw-fw+1):
                out[w,h,d] = np.sum(filter*image[w:w+fh , h:h+fw , d])  
    if id == 1:
        return np.resize(out, (out.shape[0], out.shape[1])).astype(np.uint8)
    else:
        return out.astype(np.uint8)
        

def readImg(fid):
#    ifile=ipath+str(fid).zfill(4)+'.png'
#    img = cv.imread(ifile,0)
    img = cv.imread("original.jpg")
    if len(img.shape) < 3:
        img.resize(*img.shape,1)
    return img


def showImg(img):
    if len(img.shape) < 3:
        plt.imshow(img, cmap="gray")
    elif img.shape[2] == 1:
        plt.imshow(np.resize(img,(img.shape[0],img.shape[1])), cmap="gray")
    else:
        plt.imshow(img)
        
        
def readPoint(fid):
    pfile=ppath+str(fid).zfill(4)+'.txt'
    # open point file 
    data_p=pd.read_csv(pfile,sep=' ',header=None, skiprows=[0]).to_numpy().transpose()
    return data_p

def point2img(data,r,t,w,h):
    # convert into camera coordinate frame C
    r=get_M(np.deg2rad(r),t)
    data_c=np.dot(r,data)
# convert into spherical coornidate frame and image 
    data_i=xyz2pixel(data_c,w,h)
    return data_i

def img_edge_detector(img):
#    gaussion filters
    (gfw,gfh) = (3,3)
    gaussianFilter = gaussFilter((gfw,gfh),4)
    gaussian_img = getimage_gaussianFilter(img, gaussianFilter)
    print('show gaussian')
    showImg(gaussian_img)
    
#sober operation
    print('show sober')
    img_sober =getimageedgescore_sobelOperator(gaussian_img)
#    img_sober = cv.cvtColor(img_sober, cv.COLOR_GRAY2RGB) 
    showImg(img_sober)
#   nms suppression   
    return 0

def point_edge_detector():
    return 0


#main-------------------------------
fidrange=[1,1]
r0=[-55.14, 15.47, -0.18]
t0=[0.0934, 0.0597, -0.1659]
(imgw,imgh) =(640,512) 

for  fid in np.arange(fidrange[0],fidrange[1]+1,1): 
    img=readImg(fid)
    data_p=readPoint(fid)
    data_pi=point2img(data_p,r0,t0,imgw,imgh)
    img_edge=img_edge_detector(img)
    


#est code
    
    
#    
    
    
    
    
    
