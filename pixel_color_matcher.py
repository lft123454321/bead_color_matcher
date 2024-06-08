from tkinter import filedialog as fd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import cv2 as cv

def calc_diff(c1, c2):
    return abs(int(c1[0])-int(c2[0])) + abs(int(c1[1])-int(c2[1])) + abs(int(c1[2])-int(c2[2]))

filename = fd.askopenfilename()

print(filename)

raw_img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

img = cv.fastNlMeansDenoisingColored(raw_img, None, 10, 10, 7, 15)
# img = raw_img



Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))

colors = []



for c in center:
    is_duplicated = False
    low = np.array([c[0]-5, c[1]-5, c[2]-5])
    high = np.array([c[0]+5, c[1]+5, c[2]+5])
    dst = cv.inRange(src=img, lowerb=low, upperb=high)
    xy = np.column_stack(np.where(dst==255))
    percentage = len(xy) / (img.shape[0]*img.shape[1])
    if(len(xy) == 0):
        continue
    for cc in colors:
        diff = calc_diff(cc[0], c)
        if(diff < 20):
            is_duplicated = True
    if(not is_duplicated):
        colors.append((c, xy))
        print(c, percentage)
N = len(colors)

colors = sorted(colors, key=lambda student: student[1][-1][1])

img_height = img.shape[0]

width = img.shape[1]
height = int(img.shape[0] / 5)
legend = np.zeros((height,width, img.shape[2]), dtype=np.uint8)

img_new = np.concatenate((raw_img, legend), axis = 0)

cell_width = int(width / N)
for i in range(N):
    if(i > len(colors)):
        break
    c = colors[i][0]
    start_point = (i * cell_width, img_height)
    end_point = ((i+1) * cell_width, img_height + int(height / 2))
    color = (int(c[0]), int(c[1]), int(c[2]))
    thickness = -1
    img_new = cv.rectangle(img=img_new, pt1=start_point, pt2=end_point, color=color, thickness=thickness)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img_new,str(color[0]),(i * cell_width,img_height+int(height*0.60)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(img_new,str(color[1]),(i * cell_width,img_height+int(height*0.70)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(img_new,str(color[2]),(i * cell_width,img_height+int(height*0.80)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    
    xy = colors[i][1]
    pt1 = (xy[-1][1], xy[-1][0])
    pt2 = (int((i+0.5)*cell_width), img_height)
    cv.line(img_new, pt1, pt2, (0,0,0), 1, cv.LINE_AA)

# cv.imshow('res2',res2)
# cv.imshow('img',img)
cv.imshow('img_new',img_new)
cv.waitKey(0)
cv.destroyAllWindows()