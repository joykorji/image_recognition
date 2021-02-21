#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:35:43 2021

@author Malou & Joy 
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte
from skimage import io
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage import color, morphology




orig_phantom = img_as_ubyte(data.shepp_logan_phantom())
print(orig_phantom)

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    
#get image from user
def get_image():
    img_found = False
    img = ''
    images = ['image1.pgm', 'image2.pgm', 'image3.pgm', 'image4.pgm', 'image5.pgm']
    print("Please enter which image you would like to analyze (image1, image2, image3, image4 or image5)")
    while (not img_found):  
        img = input()+'.pgm'   
        if img in images:
            img_found = True
        else:
            print("This image is not available, please enter image1, image2, image3, image4 or image5")
    return img

#get the data from pgm file
def readpgm(name):
    with open(name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # here, it makes sure it is ASCII greyscale format (P2)
    assert lines[0].strip() == 'P2' 
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    #the first two data points are the shape, the third is the max value
    #and the rest are the pixel values
    return (np.array(data[3:]),(data[1],data[0]),data[2])



#set the threshold of the image
def set_threshold(image):
    if image == 'image1.pgm' or image == 'image3.pgm':
        thresh = 220
    elif image == 'image2.pgm':
        thresh = 160
    elif image == 'image4.pgm':
        thresh = 70
    else:
        thresh = 215
    return thresh


#Connected component labeling
def connected_component_labeling(image):
    size_x, size_y = image.shape
    #negate image
    image = image * -1
    
    
    #current label
    label = 0
    
    #First pass through array to label components
    for row in range (0, size_x):
        for col in range (0, size_y):
           #if pixel is -1, assign it and its neighbors a new label
           if image[row][col] == -1:
               label += 1
               image[row][col] = label
               image = label_neighbors(image, label, row, col)
           #if pixel is already labeled, assign its neighbors its own label
           elif image[row][col] != 0:
                image = label_neighbors(image, image[row][col], row, col)
                
    return image
                

#function to label neighboring pixels
def label_neighbors(image, label, x, y):

    height, width = image.shape
    #check if not the last col
    if y != width - 1:
        if image[x][y+1] == -1:
             #check neighbor to the right
            image[x][y+1] = label
        #if the neighboor is labeled with another label, convert it to the current pixel label and save the parent in dic
        elif image[x][y+1] != label and image[x][y+1] != 0:
            Parents_tree[image[x][y+1]] = label
            image[x][y+1] = label
    #check if not the last row and col
    if x != height-1 and y != width - 1:
        #check neighbor in diagonal bellow
        if image[x+1][y+1] == -1 :
            image[x+1][y+1] = label 
            
        elif image[x+1][y+1] != label and image[x+1][y+1] != 0:
            Parents_tree[image[x+1][y+1]] = label
            image[x+1][y+1] = label
            
    #check if not the last row
    if x != height-1:
        #check neighbor bellow
        if image[x+1][y] == -1 :
            image[x+1][y] = label
        elif image[x+1][y] != label and image[x+1][y] != 0:
            Parents_tree[image[x+1][y]] = label
            image[x+1][y] = label
            
    return image


def label_merging(labels):
    
    for num in range(1, len(Parents_tree)):        #looping in all the dict 
        key_list = [] 
        if not (num in children):   #checking if the num is in the children set, if yes it means we already found the parent of this num and we should skip it
            root = num
            while(num in Parents_tree.keys()):     #checking if the num is a parent of something
                key_list.append(root)
                key_list.append(Parents_tree[num]) # if yes add the child to the list
                children.add(num)    #add the child to the set so we don't loop into it children again
                num = Parents_tree[num]            #make that child the num now so we can check if it is a parent too 
                
            list_of_parents[root] = key_list     #add root(key) with value(list) to the list_of_parents
            

def merge_overlapping_sublists(lst):
    output, refs = {}, {}
    for index, sublist in enumerate(lst):
        output[index] = set(sublist)
        for elem in sublist:
            refs[elem] = index
    changes = True
    while changes:
        changes = False
        for ref_num, sublist in list(output.items()):
            for elem in sublist:
                current_ref_num = refs[elem]
                if current_ref_num != ref_num:
                    changes = True
                    output[current_ref_num] |= sublist
                    for elem2 in sublist:
                        refs[elem2] = current_ref_num
                    output.pop(ref_num)
                    break
    return list(output.values())

def get_key(val): 
    for key, value in label_dict.items(): 
         if val in value: 
             return key 

def label_merging_final(image):
    size_x, size_y = image.shape
    
    
    #First pass through array to join same labelled components
    for row in range (0, size_x):
        for col in range (0, size_y):
           if (image[row][col] != 0 and isinstance(get_key(image[row][col]), int) ):
               image[row][col] = get_key(image[row][col])
               
                
    return image  

def clean_image(image, img_name):
    shape_x, shape_y = image.shape
    uniques = np.unique(image)
    for u in uniques:
        if np.count_nonzero(image==u) < 20:
            #np.where(image==u, 0, image)
            #[[_el if _el != u else 0 for _el in _ar] for _ar in image]
            for i in range(0, shape_x): 
                for j in range(0, shape_y):
                    if image[i][j] == u:
                        image[i][j] = 0
    return image

def condense_labels(image):
    uniques = np.unique(image)
    shape_x, shape_y = image.shape
    new_labels = dict()
    for i in range(0, len(uniques)):
        new_labels[uniques[i]] = i
    for k, v in new_labels.items(): image[image==k] = v
    return image
'''
Feature Extraction
area, C1(|P|^2/A), C2(mean/variance), second moments, bounding box
'''

#returns area of a connected component
def area(image, label):
    return np.count_nonzero(image==label)

#returns centroid as row, col tuple
def centroid(image, label):
    size_x, size_y = image.shape
    row_sum = 0
    col_sum = 0
    count = 0
    for r in range (0, size_x):
        for c in range (0, size_y):
            if image[r][c] == label:
                count += 1
                row_sum += r
                col_sum += c
    
    return row_sum/count, col_sum/count

#returns the distance between two points in a 2D plane
def distance(u, v):
    return np.sqrt((np.square(u[0]-v[0]) + np.square(v[1]-v[1])))

#returns the perimeter using 8-connectivity
def perimeter(image, label):
    shape_x, shape_y = image.shape
    #list of perimeter points
    perimeter = []
    for x in range (0, shape_x):
        for y in range (0, shape_y):
            if image[x][y] == label:
                #check if point is in perimeter
                if is_perimeter(image, x, y):
                    perimeter.append([x,y])
    return perimeter

#returns true if a point is in the perimeter of a region, false otherwise (using 8 connectivity)           
def is_perimeter(image, x, y):
    shape_x, shape_y = image.shape
    for i in range (x-1, x+2):
        for j in range (y-1, y+2):
            if i < shape_x and i > -1 and j < shape_y and j > -1:
                if image[i][j] == 0:
                    return True
    return False

def perimeter_length(perimeter):
    return len(perimeter)

def circularity_1(perimeter_length, area):
    return np.round(perimeter_length * perimeter_length/area, 2)
    
def circularity_2(perimeter, centroid):
    mean_total = 0
    variance_total = 0
    count = 0
    row, col = centroid
    for pt in perimeter:
        mean_total += distance(pt, [row, col])
        count += 1
    mean = mean_total/count
    #print("count: ", count)
    #print("mean total ", mean_total)
    #print("mean: ", mean)
    for pt in perimeter:
        variance_total += np.square(distance(pt, [row, col]) - mean)
    variance = variance_total/count
    #print("variace total: ", variance_total)
    #print("variance: ", variance)
    return np.round(mean/variance, 2)
'''
#array from lecture 2 ppt
c_test = np.array([[0,0,0,0,0,0,0,0], [0,0,0,1,1,0,0,0],
                    [0,0,0,1,1,1,1,1], [0,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,0,0], [0,0,0,1,1,0,0,0],
                    [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0]]) 

perim = perimeter(c_test, 1)
perim.remove([2,4])
perim.remove([3,3])
perim.remove([3,5])
perim.remove([4,3])
perim.remove([4,4])
center = centroid(c_test, 1)
c2 = circularity_2(perim, center)
print(c2)
'''    
def second_moment_row(image, label, centroid):
    r, c = centroid
    moment_total = 0
    count = 0
    shape_x, shape_y = image.shape
    for x in range(0, shape_x):
        for y in range(0, shape_y):
            if image[x][y] == label:
                moment_total += np.square(x-r)
                count += 1
    return np.round(moment_total/count, 2)
            
def second_moment_col(image, label, centroid):
    r, c = centroid
    moment_total = 0
    count = 0
    shape_x, shape_y = image.shape
    for x in range(0, shape_x):
        for y in range(0, shape_y):
            if image[x][y] == label:
                moment_total += np.square(y-c)
                count += 1
    return np.round(moment_total/count, 2)

def second_moment_mixed(image, label, centroid):
    r, c = centroid
    moment_total = 0
    count = 0
    shape_x, shape_y = image.shape
    for x in range(0, shape_x):
        for y in range(0, shape_y):
            if image[x][y] == label:
                moment_total += (x-r) * (y-c)
                count += 1
    return np.round(moment_total/count, 2)


def bounding_box(image, label, box_label):
    shape_x, shape_y = image.shape
    left = shape_x-1
    right = 0
    top = shape_y-1
    bottom = 0
    #get corners of bounding box
    for i in range (0, shape_x):
        for j in range (0, shape_y):
            if image[i][j] == label:
                if i < left:
                    left = i
                if i > right: 
                    right = i
                if j < top:
                    top = j
                if j > bottom:
                    bottom = j
    #draw bounding box 
    for i in range(left-1, right+2):
        if i > -1 and i < shape_x:
            if top-1 > -1:
                if image[i][top-1] == 0:
                    image[i][top-1] = box_label
            if bottom+1 < shape_y:
                if image[i][bottom+1] == 0:
                    image[i][bottom+1] = box_label
    for j in range(top-1, bottom+2):
        if j > -1 and j < shape_y:
            if left-1 > -1:
                if image[left-1][j] == 0:
                    image[left-1][j] = box_label
            if right+1 < shape_x:
                if image[right+1][j] == 0:
                    image[right+1][j] = box_label
    return image

def output(image, labels):
    #area, C1(|P|^2/A), C2(mean/variance), second moments, bounding box
    bounding_box_label = np.max(labels) + 1
    for label in labels:
        if label !=0:
            print("Connected Componenent", label)
            cc_area = area(image, label)
            print("\tArea:", cc_area)
            cc_perimeter = perimeter(image, label)
            print("\tC1:", circularity_1(perimeter_length(cc_perimeter), cc_area))
            cc_centroid = centroid(image, label)
            print("\tC2:", circularity_2(cc_perimeter, cc_centroid))
            print("\tSecond moment row:", second_moment_row(image, label, cc_centroid))
            print("\tSecond moment column:", second_moment_col(image, label, cc_centroid))
            print("\tSecond moment mixed:", second_moment_mixed(image, label, cc_centroid))
            image = bounding_box(image, label, bounding_box_label)
    plt.figure()
    plt.title("Bounding Box")
    plt.imshow(image)
            

'''pipeline'''
img = get_image()    
pgm = readpgm(img)
#reshape image as a 2D array
data = np.reshape(pgm[0],pgm[1])
#plt.imshow(data)
size_x, size_y = pgm[1]
#threshold image
thresh = (data < set_threshold(img)).astype(int)
#plt.imshow(thresh)
#start connected component labelling
Parents_tree = {}
labels = connected_component_labeling(thresh)             
list_of_parents = {}
children = set()
label_merging(labels)
list_of_lists = list_of_parents.values()
label_parents = merge_overlapping_sublists(list_of_lists)
label_parents_tree = [x for x in label_parents if x] # remove empty sets from list_of_lists
label_dict = {}
i = 1
for list in label_parents_tree:    #assigning a label to each list 
    label_dict[i] = list
    i= i+1
labelled = label_merging_final(labels) #this is the labeled components 
labelled = clean_image(labelled, img)
labelled = condense_labels(labelled)
labels = np.unique(labelled)
plt.figure()
plt.title('Connected Compononets')
plt.imshow(labelled)
output(labelled, labels)


if img == "image5.pgm":
    
    selem = morphology.disk(1)
    
    selem1 = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1] ])
    
    
    dilated = dilation(thresh, selem1)
    plot_comparison(thresh, dilated, 'dilation')
    
    eroded = erosion(dilated, selem)
    plot_comparison(dilated, eroded, 'erosion')
    
    eroded2 = erosion(eroded, selem)
    plot_comparison(eroded, eroded2, 'erosion')
    
    
    



      
