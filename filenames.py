# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 00:49:03 2020

@author: mohit123
"""

import os

from shutil import copyfile
import scipy.misc
import matplotlib

folder_path = "Leaf/test1"
#folder_path1 = "C:/Mohit/2020/Oct20/COVID/COVID/COVID50/(1) JPG"
# Specify the output jpg/png folder path

images_path = os.listdir(folder_path)
index = 0


jpg_files = []
dest_files = []

file1 = open('test1.txt', 'w')
for n, image in enumerate(images_path):
    src = os.path.join(folder_path, image)
    jpg_files.append(src)
    
    file1.write(image)
    file1.write('\n')
file1.close()