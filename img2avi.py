#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:57:09 2020

@author: guoxiong
"""

import cv2
import glob

out = cv2.VideoWriter("input.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20.0, 2048, 416)

while True:
    for name in glob.glob("./test_data/test_2048_832_3/*.jpg"):
        img = cv2.imread(name)
        out.write(img)
        
out.release()
