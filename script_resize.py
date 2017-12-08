import numpy as np
import cv2
import os
j=1
print('input the absolute address of the directory')
directory=input()
name=input('name')
os.chdir('/home/dristibutola/datacollected/'+str(directory))
ndir=input('new dir')
k=os.listdir()
for i in k:
    im=cv2.imread(i)
    im=cv2.resize(im,(180,200))
    os.chdir('/home/dristibutola/datacollected/'+str(ndir))
    cv2.imwrite(str(name)+'.'+str(j)+'.jpg',im)
    j+=1


