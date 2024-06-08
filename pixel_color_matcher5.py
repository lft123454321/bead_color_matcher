from tkinter import filedialog as fd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import skimage
import cv2 as cv

import csv

from tqdm import tqdm

cell_rows = 50
cell_cols = 50

def calc_diff_rgb(c1, c2):
    return (int(c1[0])-int(c2[0]))**2 + (int(c1[1])-int(c2[1]))**2 + (int(c1[2])-int(c2[2]))**2

def calc_diff_hsv(c1, c2):
    bgr = np.uint8([[[c1[0],c1[1],c1[2] ], [c2[0],c2[1],c2[2] ]]])
    hsv = cv.cvtColor(bgr,cv.COLOR_BGR2HSV)
    h2h1 = (int(hsv[0][0][0]) - int(hsv[0][1][0]))
    if(h2h1 < -127):
        h2h1 += 256
    if(h2h1 > 127):
        h2h1 -= 256
    h2h1 = h2h1 ** 2
    s2s1 = (int(hsv[0][0][1]) - int(hsv[0][1][1])) ** 2
    v2v1 = (int(hsv[0][0][2]) - int(hsv[0][1][2])) ** 2
    if(hsv[0][0][1] < 10 and hsv[0][1][1] < 10):
        h2h1 = 0        
    if(hsv[0][0][2] < 10 and hsv[0][1][2] < 10):
        h2h1 = 0

    return h2h1*3 + s2s1*1 + v2v1*2

def calc_diff(c1, c2):
    return calc_diff_hsv(c1, c2) + calc_diff_rgb(c1, c2) * 2

def find_nearest_color(c, palette):
    nearest_diff = 255 * 255
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
csv_dict = {}

csv_filename = fd.askopenfilename()

print(csv_filename)
with open(csv_filename, newline='', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(row)
        c = (row['COCO'], int(row['B']), int(row['G']),int(row['R']))
        csv_palette.append(c)
        csv_dict[row['COCO']] = (c[1], c[2], c[3])


# img = skimage.io.imread(filename)
raw_img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

img = raw_img.copy()

img_new = np.zeros_like(img)

cell_width = img.shape[1] / cell_cols
cell_height = img.shape[0] / cell_rows

bins = {}

for i in tqdm(range(cell_rows)):
    start_y = int(cell_height * i)
    end_y = int(cell_height * (i+1))
    text_start_y = int(cell_height * (i+0.7))
    for j in range(cell_cols):
        start_x = int(cell_width * j)
        end_x = int(cell_width * (j+1))
        roi = img[start_y:end_y, start_x:end_x]
        flat = roi.reshape((roi.shape[0]*roi.shape[1], 3))
        median = np.median(flat, axis=0)
        nearest = find_nearest_color(median, csv_palette)
        nearest_name = nearest[0]
        color = csv_dict[nearest_name]

        pt1 = (start_x, start_y)
        pt2 = (end_x, end_y)
        img_new = cv.rectangle(img=img_new, pt1=pt1, pt2=pt2, color=color, thickness=-1)

        font = cv.FONT_HERSHEY_PLAIN
        text_color = (0,0,0)
        if(color[0]+color[1]+color[2] < 128*3):
            text_color = (255,255,255)
        pt_text = (start_x, text_start_y)
        cv.putText(img, nearest_name, pt_text, font, cell_height/32, text_color, 1, cv.LINE_AA)


        if(nearest_name not in bins):      
            bins[nearest_name] = 1
        else:
            bins[nearest_name] += 1

for i in range(1, cell_rows):
    start_x = 0
    end_x = img.shape[1]
    start_y = int(i * cell_height)
    end_y = start_y
    pt1 = (start_x, start_y)
    pt2 = (end_x, end_y)
    cv.line(img, pt1, pt2, (127,127,127), 1, cv.LINE_AA)
for i in range(1, cell_cols):
    start_x = int(i * cell_width)
    end_x = start_x
    start_y = 0
    end_y = img.shape[0]
    pt1 = (start_x, start_y)
    pt2 = (end_x, end_y)
    cv.line(img, pt1, pt2, (127,127,127), 1, cv.LINE_AA)

N = len(bins)
legend = np.zeros((img.shape[0], int(img.shape[1]/5),img.shape[2]), dtype=np.uint8)

img = np.concatenate((img, legend), axis = 1)

img_width = raw_img.shape[1]
legend_width = int(raw_img.shape[0] / 5)
legend_cell_height = raw_img.shape[0] / N
i = 0
for nearest_name in bins:
    start_point = (img_width, int(i * legend_cell_height))
    end_point = (img_width + int(img_width / 5), int((i+1) * legend_cell_height))
    color = csv_dict[nearest_name]
    img = cv.rectangle(img=img, pt1=start_point, pt2=end_point, color=color ,thickness=-1)
    font = cv.FONT_HERSHEY_SIMPLEX
    text_color = (0,0,0)
    if(color[0]+color[1]+color[2] < 128*3):
        text_color = (255,255,255)
    cv.putText(img, nearest_name, (img_width, int((i+0.7) * legend_cell_height)), font, 0.5, text_color, 1, cv.LINE_AA)
    cv.putText(img, str(bins[nearest_name]), (img_width+int(legend_width*0.5), int((i+0.7) * legend_cell_height)), font, 0.5, text_color, 1, cv.LINE_AA)
    i += 1

cv.imshow('raw',raw_img)
cv.imshow('labeled',img)
cv.imshow('simulation',img_new)
cv.waitKey(0)
cv.destroyAllWindows()
exit()


bins = {}

for i in tqdm(range(img.shape[0])):
    for j in range(img.shape[1]):
        c = img[i][j]
        c = (int(c[0]), int(c[1]), int(c[2]))
        if(c not in bins):      
            bins[c] = {'count':1, 'pos':[(j,i)]}
        else:
            bins[c]['count'] += 1
            bins[c]['pos'].append((j,i))

tuple_bins = []

for bin in bins:
    nearest_name = bin
    count = bins[nearest_name]['count']
    pos = bins[nearest_name]['pos']
    tuple_bins.append((nearest_name, count, pos))

sorted_bins = sorted(tuple_bins, key=lambda b: b[1], reverse=True)

img_height = img.shape[0]

width = img.shape[1]
height = int(img.shape[0] / 5)
legend = np.zeros((height,width, img.shape[2]), dtype=np.uint8)

img_labels = np.zeros_like(raw_img)

img_new = np.concatenate((raw_img, legend), axis = 0)

N = len(sorted_bins)
MAX_N = 500


filtered_bins = {}

for i in range(N):
    count = sorted_bins[i][1]
    if(count < 2):
        break
    if(i > MAX_N):
        break
    color = sorted_bins[i][0]
    pos = sorted_bins[i][2]
    nearest = find_nearest_color(color, csv_palette)
    nearest_name = nearest[0]
    if(nearest_name not in filtered_bins):
        filtered_bins[nearest_name] = pos
    
N = len(filtered_bins)

cell_width = width / N
i = 0
for bin in filtered_bins:
    nearest_name = bin
    start_point = (int(i * cell_width), img_height)
    end_point = (int((i+1) * cell_width), img_height + int(height / 2))
    color = csv_dict[nearest_name]
    thickness = -1
    img_new = cv.rectangle(img=img_new, pt1=start_point, pt2=end_point, color=color, thickness=thickness)
    font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(img_new,nearest_name,(int(i * cell_width),img_height+int(height*0.60)), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(img_new,nearest_name,(filtered_bins[bin][0]), font, 0.3,(0,0,0),1,cv.LINE_AA)

    i += 1

cv.imshow('img_new',img_new)
cv.waitKey(0)
cv.destroyAllWindows()