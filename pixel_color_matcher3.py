from tkinter import filedialog as fd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import skimage
import cv2 as cv

import csv

def calc_diff(c1, c2):
    return abs(int(c1[0])-int(c2[0])) + abs(int(c1[1])-int(c2[1])) + abs(int(c1[2])-int(c2[2]))

def find_nearest_color(c, palette):
    nearest_diff = 255 * 3
    nearest_color = None
    for p in palette:
        d = calc_diff(c, (p[1], p[2], p[3]))
        if(d < nearest_diff):
            nearest_diff = d
            nearest_color = p
    return nearest_color

def approximately_in(c, bins):
    nearest_diff = 255 * 3
    bin = None
    for b in bins:
        d= calc_diff(c, b[0])
        if(d < nearest_diff):
            nearest_diff = d
            bin = b
    if(nearest_diff < 10):
        return True, bin
    else:
        return False, None

filename = fd.askopenfilename()

print(filename)

csv_palette = []

csv_filename = fd.askopenfilename()

print(csv_filename)
with open(csv_filename, newline='', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)
        c = (row['COCO'], int(row['B']), int(row['G']),int(row['R']))
        csv_palette.append(c)



# img = skimage.io.imread(filename)
raw_img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

img = raw_img

bins = {}

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        c = img[i][j]
        c = (int(c[0]), int(c[1]), int(c[2]))
        if(c not in bins):      
            bins[c] = {'count':1, 'pos':[(i,j)]}
        else:
            bins[c]['count'] += 1
            bins[c]['pos'].append((i,j))

tuple_bins = []

for bin in bins:
    c = bin
    count = bins[c]['count']
    pos = bins[c]['pos']
    tuple_bins.append((c, count, pos))

sorted_bins = sorted(tuple_bins, key=lambda b: b[1], reverse=True)

img_height = img.shape[0]

width = img.shape[1]
height = int(img.shape[0] / 5)
legend = np.zeros((height,width, img.shape[2]), dtype=np.uint8)

img_labels = np.zeros_like(raw_img)

img_new = np.concatenate((raw_img, legend), axis = 0)

N = len(sorted_bins)
MAX_N = 500
for i in range(MAX_N):
    print(sorted_bins[i][0])
    print(sorted_bins[i][1])

filtered_bins = []
for i in range(N):
    count = sorted_bins[i][1]
    if(count < 2):
        break
    if(i > MAX_N):
        break
    matched, c = approximately_in(sorted_bins[i][0], filtered_bins)
    if not matched:
        filtered_bins.append(sorted_bins[i])

    
N = len(filtered_bins)

cell_width = width / N
for i in range(N):
    if(i > len(filtered_bins)):
        break
    c = filtered_bins[i][0]
    start_point = (int(i * cell_width), img_height)
    end_point = (int((i+1) * cell_width), img_height + int(height / 3))
    color = (int(c[0]), int(c[1]), int(c[2]))
    thickness = -1
    img_new = cv.rectangle(img=img_new, pt1=start_point, pt2=end_point, color=color, thickness=thickness)
    font = cv.FONT_HERSHEY_SIMPLEX
    # cv.putText(img_new,str(color[0]),(i * cell_width,img_height+int(height*0.60)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    # cv.putText(img_new,str(color[1]),(i * cell_width,img_height+int(height*0.70)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    # cv.putText(img_new,str(color[2]),(i * cell_width,img_height+int(height*0.80)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    
    # xy = filtered_bins[i][2]
    # pt1 = (xy[100][1], xy[100][0])
    # pt2 = (int((i+0.5)*cell_width), img_height)
    # cv.line(img_new, pt1, pt2, (0,0,0), 1, cv.LINE_AA)

    # interted_c = (255-c[0], 255-c[1], 255-c[2])
    # cv.putText(img_new, str(color[2])+','+str(color[1])+','+str(color[0]), (xy[0][1], xy[0][0]+10), font, 0.3, interted_c, 1, cv.LINE_AA)

    nearest_color = find_nearest_color(color, csv_palette)
    if nearest_color is None:
        continue
    start_point = (int(i * cell_width), img_height + int(height / 3))
    end_point = (int((i+1) * cell_width), img_height + int(height * 2 / 3))
    matched_color = (nearest_color[1], nearest_color[2], nearest_color[3])
    matched_name = nearest_color[0]
    img_new = cv.rectangle(img=img_new, pt1=start_point, pt2=end_point, color=matched_color, thickness=thickness)
    cv.putText(img_new,matched_name,(int(i * cell_width),img_height+int(height*0.80)), font, 0.3,(255,255,255),1,cv.LINE_AA)


cv.imshow('img_new',img_new)
cv.waitKey(0)
cv.destroyAllWindows()