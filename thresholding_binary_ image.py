#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:35:43 2021

@author: Malou & Joy 
"""

import numpy as np
import matplotlib.pyplot as plt



#get image from user
def get_image():
    img_found = False
    img = ''
    images = ['image1.pgm', 'image2.pgm', 'image3.pgm', 'image4.pgm', 'image5.pgm']
    print("\n Hello, \n \n This Program asks the user to enter an image and then will analysis it. \n After that it will display information about this picture (the information includes features of the image and number of connected component in it. ")
    print("\n Please enter which image you would like to analyze (image1, image2, image3, image4 or image5)")
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
    elif image == 'image4.pgm':
        thresh = 70
    elif image == 'image3.pgm':
        thresh = 190
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









        
      
        
      
        
      
        
      
        
      
        
      
        