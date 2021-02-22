#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:35:43 2021

@author: Malou Merovitz & Joy Korji
Connected component analysis program
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, erosion, disk


#get image from user
def get_image():
    img_found = False
    img = ''
    images = ['image1.pgm', 'image2.pgm', 'image3.pgm', 'image4.pgm', 'image5.pgm']
    
    print("\n Hello, please enter your name.")
    user = input()
    doctor=""
    if user == "layachi":
        doctor = "Dr."
    print("\n Hello, \n \n This Program asks the user to enter an image and then will analysis it. \n After that it will display information about this picture (the information includes features of the image and connected component images. ")
    print("\n We promise this is the last time we ask for input, Please enter which image you would like to analyze (image1, image2, image3, image4 or image5)")
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
    if image == 'image1.pgm':
        thresh = 220
    elif image == 'image2.pgm':
        thresh = 160
    elif image == 'image3.pgm':
        thresh = 230
    elif image == 'image4.pgm':
        thresh = 70
    else:
        thresh = 215
    return thresh

'''
Connected component analysis
'''
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

#This function will built the union structure (here we are using a dictionary)
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
            
#will merge all the parent-child in the dictionary with the root (the first parent)
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

#gets the key of a specific value
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

#Cleans out the images, and apply the morphological filtering for some images.
def clean_image(image, img_name):
    shape_x, shape_y = image.shape
    uniques = np.unique(image)
    if img_name != 'image3.pgm':
        #clean out components with less than 20 pixels
        clean = []
        for u in uniques:
            if np.count_nonzero(image==u) < 20:
                clean.append(u)
        for c in clean:
            for i in range(0, shape_x): 
                for j in range(0, shape_y):
                    if image[i][j] == c:
                        image[i][j] = 0  
    if img_name == "image5.pgm": 
        #apply morphological filtering
        selem = disk(1)
        selem1 = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1] ])
        dilated = dilation(image, selem1)
        eroded = erosion(dilated, selem)
        image = erosion(eroded, selem)

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
area, C1, C2, second moments, bounding box
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

#returns the c1 value of the connected component
def circularity_1(perimeter_length, area):
    return np.round(perimeter_length * perimeter_length/area, 2)

#returns the c2 value of the connected component    
def circularity_2(perimeter, centroid):
    mean_total = 0
    variance_total = 0
    count = 0
    row, col = centroid
    for pt in perimeter:
        mean_total += distance(pt, [row, col])
        count += 1
    mean = mean_total/count
    for pt in perimeter:
        variance_total += np.square(distance(pt, [row, col]) - mean)
    variance = variance_total/count
    return np.round(mean/variance, 2)
 
#second moment functions return second moment row/col/mixed of connected component
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

#Draws bounding box around connected component
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

#Outputs all the features and plots
def output(image, labels):
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
#threshold image
thresh = (data < set_threshold(img)).astype(int)
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
#output labelled connected component image
plt.figure()
plt.title('Connected Compononets')
plt.imshow(labelled)
#output features
output(labelled, labels)
