#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import sys
import math
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import fingerprint_enhancer
from skimage.morphology import skeletonize
from scipy.spatial import ConvexHull
from numpy.random import randint
from numpy.random import rand

# In[2]:


class Preprocess:
    GLOBAL_THRESHOLD = 0.01

    def __init__(self):
        pass
#         self.image = image
    def area(self,image):
        ar,im = self.segmentation(image,10)
        return ar
        
    def segmentation(self,image,block_size=15):
        n,m = image.shape
        new_image = np.zeros((n,m))
        area = 0
        def mean_grey_level(image_block,block_size):   
            return np.sum(image_block)/(block_size*block_size)

        def grey_scale_variance(image_block,block_size):
            mean_val = mean_grey_level(image_block,block_size)
            return np.sum(np.square(image_block - mean_val))/(block_size**2)

        for row in range((n//block_size)+1):
            for col in range((m//block_size)+1):
                block  = image[row*block_size:(row+1)*block_size,col*block_size:(col+1)*block_size]
                var = grey_scale_variance(block,block_size)
                
                if var < Preprocess.GLOBAL_THRESHOLD:
                    new_image[row*block_size:(row+1)*block_size,col*block_size:(col+1)*block_size] = np.zeros(block.shape)
                else:
                    new_image[row*block_size:(row+1)*block_size,col*block_size:(col+1)*block_size] = block
                    area+=block_size*block_size
        
        return area,new_image    
    
    def normalization(self,image):
        n,m = image.shape

        M_0 = 0.5
        V_0 = 0.5
        M = image.mean()
        V = np.sum(np.square(image - M))/(n*m)
    #     print(M,V)

        temp =  np.sqrt((V_0/V) * np.square(image - M))
        N = M_0 - temp
        N = N + np.multiply(2*(image > M),temp)

        return N
    def enhancement(self,image):
        out = fingerprint_enhancer.enhance_Fingerprint(image)
        return out

    def binarization(self,image):
        ret,thresh1 = cv2.threshold(image,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh1
    
    def thinning(self,image):
        image = skeletonize(image)
        return image
    
    def run_all(self,image):
        _,image = self.segmentation(image)
        image = self.normalization(image)
        image = self.enhancement(image)
        image = self.binarization(image)
        image = (image)//255
        image = self.thinning(image)
        image = 1 - image
        return image


# In[3]:


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


# In[4]:


def orientation_field(Gx,Gy, N):
    (b, l) = Gx.shape
    target_l = l-N+1
    target_b = b-N+1
    mx = 0
    
#     Intialising the matrix with null entries
    orientation_matrix = np.zeros(shape=(target_b,target_l))

    for i in range(target_b):
        for j in range(target_l):
#           Get the current matrix
            mat_x = Gx[i:i+N,j:j+N]
            mat_y = Gy[i:i+N,j:j+N]
            
            num = np.sum(np.multiply(mat_x,mat_y))*2
            denom = np.sum(np.multiply(mat_x,mat_x)-np.multiply(mat_y,mat_y))
#             print(denom)
#             orientation_matrix[i,j] = math.degrees(math.atan(num/denom))/2 + (math.pi)/2
            orientation_matrix[i,j] = math.degrees(math.atan(num/denom))/2 + 90
    
    return orientation_matrix


# In[5]:


def minutiae_extract(image,orientation_matrix):
#     print(image)
#     plt.imshow(image,cmap="gray")
    n,m = image.shape
    minutae_set = []
    for i in range(1,n-1):
        for j in range(1,m-1):
            block = image[i-1:i+2,j-1:j+2]
#             print(i,j,block)
            if block[1][1] == 0:
                temp = block.sum()
#                 print("Block Sum:",temp)
                if temp == 8 or temp == 7:
                    element = (i,j,orientation_matrix[i-1][j-1])
                    minutae_set.append(element)
    return minutae_set


# In[6]:


'''
    Hough Transform. Maybe del_theta because being float not be exactly equal. Add a epsilon for del_theta if required.
'''
def hough_transform(Q_set, T_set):
    A ={}
    max_val, max_val_key = -float('inf'),None
    for i in range(len(Q_set)):
        for j in range(len(T_set)):
            x_q,y_q,theta_q = Q_set[i]
            x_t,y_t,theta_t = T_set[j]
            
            del_theta = theta_t - theta_q
            del_x = x_t - x_q * math.cos(del_theta) - y_q * math.sin(del_theta)
            del_y = y_t + x_q * math.sin(del_theta) - y_q * math.cos(del_theta)
            k = (int(del_theta),int(del_x),int(del_y))
            if k not in A:
                A[k] = 1
            else:
                A[k] +=1
            if A[k] > max_val:
                max_val = A[k]
                max_val_key = k
#     print("Dictionary:",A)
    return max_val_key


# In[7]:


'''
    Alignment operator

'''
def align(query_set,del_theta,del_x,del_y):
    aligned_query = []
    for current_query in query_set:
        x,y,theta = current_query
        new_theta = theta + del_theta
        new_x = x + del_x * math.cos(new_theta) - del_y * math.sin(new_theta)
        new_y = y + del_y * math.cos(new_theta) + del_x * math.sin(new_theta)
        aligned_query.append((new_x,new_y,new_theta))
    return aligned_query


# In[8]:


DISTANCE_THRESHOLD = 10
ROTATION_THRESHOLD = 20

def minutiae_pairing(Q_set,T_set,transform_params):
    del_x,del_y,del_theta = transform_params
    f_T = [False]*len(T_set)
    f_Q = [False]* len(Q_set)
    count= 0
    ret = []
    
#     aligned_Q_set = align(Q_set,del_theta,del_x,del_y)
    aligned_Q_set = Q_set
#     aligned_T_set = T_set
    
    for i in range(len(Q_set)):
        Qx, Qy, Qtheta = aligned_Q_set[i]
        for j in range(len(T_set)):
            if (not f_T[j] and not f_Q[i]):
                new_Tx,new_Ty, new_Ttheta = T_set[j]
                new_del_theta =  new_Ttheta - Qtheta
                new_del_x = new_Tx - (Qx * math.cos(new_del_theta)) - (Qy * math.sin(new_del_theta))
                new_del_y = new_Ty + (Qx * math.sin(new_del_theta)) - (Qy * math.cos(new_del_theta))
                d_i_j =  math.sqrt(new_del_x**2 + new_del_y**2)
#                 print("Distance:",d_i_j,"Rotation:",abs(new_del_theta))
                if d_i_j < DISTANCE_THRESHOLD and  abs(new_del_theta) < ROTATION_THRESHOLD:
                    f_T[j] = True
                    f_Q[i] = True
                    count+=1
                    ret.append((j,i))
    return ret
    


# In[33]:


def match_score(matched_minutiae,T_set,Q_set,image1_area,image2_area):
#     MATCH_THRESHOLD = min(20,len(T_set)//2)
    MATCH_THRESHOLD = 10
    
    T_points = []
    Q_points = []
    for i,j in matched_minutiae: #i in T_set, j in Q_set
        T_points.append((T_set[i][0],T_set[i][1]))
        Q_points.append((Q_set[j][0],Q_set[j][1]))
#     if min(len(T_points),len(Q_points)) >= 3:
#         hull_T = ConvexHull(T_points)
#         hull_Q = ConvexHull(Q_points)
#         area_T = hull_T.volume
#         area_Q = hull_Q.volume

#         feature_2 = min((area_T/image1_area) , (area_Q/image2_area))
#     else:
#         feature_2 = 0
    
#     feature_1 = len(matched_minutiae)/min(len(T_set),len(Q_set))
    feature_1 = len(matched_minutiae)
    feature_2 = 0
    
#     print("Match Score:",0.9*feature_1+0.1*feature_2)
    return 0.9*feature_1+0.1*feature_2 >=MATCH_THRESHOLD 
    


# In[ ]:


def fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold):
    #     Converting degrees to radian
    theta = math.radians(theta) 
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    num = 0

    for i in range(len(mx)):
        for j in range(i,len(qx)):
            px = s*(mx[i]*cos_theta - my[i]*sin_theta) - qx[j] + tx
            py = s*(mx[i]*sin_theta + my[i]*cos_theta) - qy[j] + ty

            if(math.sqrt(px*px + py*py)< thold):
                num = num + 1
    
    return num

def convert_list_to_integer(lst):
    sum = 0
    base = 1
    for i in range(len(lst)):
        sum = sum + base*lst[i]
        base = base*2
        
    return sum

def get_value_from_chromosome(single_chromo):
    #     first 5 bits represents scale
    s_list = single_chromo[:5]
#     Next 6 bits represents angle
    theta_list = single_chromo[5:11]
#     Next 8 bits represents movement in x axis
    tx_list = single_chromo[11:19]
#     Next 8 bits represents movement in y axis
    ty_list = single_chromo[19:27]
    
    s = convert_list_to_integer(s_list)*0.01 + 0.9
    theta = convert_list_to_integer(theta_list) - 30
    tx = convert_list_to_integer(tx_list)-128
    ty = convert_list_to_integer(ty_list)-128
    
    return s, theta, tx, ty

def Random_population_generation(n_pop):
    #     n_pop is the total number of random population needed generally n_pop = total number of minutiae of query set.
    pop = [randint(0, 2, 27).tolist() for _ in range(n_pop)]
    return pop

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

#             best, score = genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy)
def genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy):

    # mx, my, tx, ty are list of numbers. mx, my are for template and qx, qy are for query
    
    # initial population of random bitstring
    pop = [randint(0, 2, 27).tolist() for _ in range(n_pop)]
    
    # keep track of best solution
    s, theta, tx, ty = get_value_from_chromosome(pop[0])
#     print(s, theta, tx, ty)
    best, best_eval = 0, fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold)
    # enumerate generations
    for gen in range(n_iter):
        
        # evaluate all candidates in the population
        scores = []
        for i in range(len(pop)):
            s, theta, tx, ty = get_value_from_chromosome(pop[i])
            scores.append(fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold))
        
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
                
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        
    return [best, best_eval]

# Hyperparameters like thold and final_thresh needs to be tunned for best possible result
def Genetic(minutiae_set1, minutiae_set2):
    mx = []
    my = []
    for i in range(len(minutiae_set1)):
        mx.append(minutiae_set1[i][0])
        my.append(minutiae_set1[i][1])

    qx = []
    qy = []
    for i in range(len(minutiae_set2)):
        qx.append(minutiae_set2[i][0])
        qy.append(minutiae_set2[i][1])

    # define the total iterations
    n_iter = 100

    # define the population size. This varries as the number of miniuate of images varries
    n_pop = 100

    # crossover rate
    r_cross = 0.9

    # mutation rate
    r_mut = 1.0 / (float(27) * 2)

    thold = 100
    final_thres = 50

    # perform the genetic algorithm search
    best, score = genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy)

    if(score>final_thres):
        return True
    else:
        return False


# In[10]:


def cache_training_set(train_set):
    d = {}

    preprocess_obj = Preprocess()
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    for filename,image in train_set:
        name = filename.split("_")[0]
        preprocessed_image = preprocess_obj.run_all(image)
        image_area = preprocess_obj.area(image)
        G_x = convolve2D(image, kernel_x,padding=0)
        G_y = convolve2D(image, kernel_y,padding=0)
        temp = orientation_field(G_x, G_y, 5)
        minutae_set = minutiae_extract(preprocessed_image,temp)
        
        if name in d:
            d[name].append((image_area,minutae_set))
        else:
            d[name] = [(image_area,minutae_set)]
    return d


# In[28]:


def cached_option_1(training_minutiae_set, test_set):
    preprocess_obj = Preprocess()
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    results = []
    for filename, image in test_set:
        name = filename.split("_")[0]
        preprocessed_image = preprocess_obj.run_all(image)
        image_area = preprocess_obj.area(image)
        G_x = convolve2D(image, kernel_x,padding=0)
        G_y = convolve2D(image, kernel_y,padding=0)
        temp = orientation_field(G_x, G_y, 5)
        minutae_set = minutiae_extract(preprocessed_image,temp)
        
        d = {}
        for k in training_minutiae_set:
            for im_area,im1 in training_minutiae_set[k]:
                matches = minutiae_pairing(minutae_set,im1,(None,None,None))
                is_a_match = match_score(matches,im1,minutae_set,image_area,im_area)
                results.append([k == name,is_a_match])
    return results
    


# In[27]:


def cached_option_2(training_minutiae_set, test_set):
    preprocess_obj = Preprocess()
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    results = []
    for filename, image in test_set:
        name = filename.split("_")[0]
        preprocessed_image = preprocess_obj.run_all(image)
        image_area = preprocess_obj.area(image)
        G_x = convolve2D(image, kernel_x,padding=0)
        G_y = convolve2D(image, kernel_y,padding=0)
        temp = orientation_field(G_x, G_y, 5)
        minutae_set = minutiae_extract(preprocessed_image,temp)
        
        for k in training_minutiae_set:
            for a,l in training_minutiae_set[k]:
                boo = Genetic(minutae_set,l)
                results.append([k == name,boo])
            
    return results


# In[13]:


# tr,te = generate_dataset("./DB1",4,1)
# d = cache_training_set(tr)
# cached_option_1(d,te)


# In[14]:


def option1(image1,image2):
    preprocess_obj = Preprocess()
    preprocessed_image1 = preprocess_obj.run_all(image1)
    preprocessed_image2 = preprocess_obj.run_all(image2)
    
    image1_area = preprocess_obj.area(image1)
    image2_area = preprocess_obj.area(image2)
    
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    G_x1 = convolve2D(image1, kernel_x,padding=0)
    G_y1 = convolve2D(image1, kernel_y,padding=0)
    G_x1 = preprocess_obj.run_all(G_x1)
    G_y1 = preprocess_obj.run_all(G_y1)
    temp1 = orientation_field(G_x1, G_y1, 5)
    G_x2 = convolve2D(image2, kernel_x,padding=0)
    G_y2 = convolve2D(image2, kernel_y,padding=0)
    G_x2 = preprocess_obj.run_all(G_x2)
    G_y2 = preprocess_obj.run_all(G_y2)
    temp2 = orientation_field(G_x2, G_y2, 5)
    
    minutae_set1 = minutiae_extract(preprocessed_image1,temp1)
    minutae_set2 = minutiae_extract(preprocessed_image2,temp2)
    transform_params = hough_transform(minutae_set1,minutae_set2)
    matches = minutiae_pairing(minutae_set1,minutae_set2,transform_params)
    is_a_match = match_score(matches,minutae_set2,minutae_set1,image2_area,image1_area)
    return is_a_match


# In[24]:


def option2(image1,image2):
    preprocess_obj = Preprocess()
    preprocessed_image1 = preprocess_obj.run_all(image1)
    preprocessed_image2 = preprocess_obj.run_all(image2)
    
    image1_area = preprocess_obj.area(image1)
    image2_area = preprocess_obj.area(image2)
    
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    G_x1 = convolve2D(image1, kernel_x,padding=0)
    G_y1 = convolve2D(image1, kernel_y,padding=0)
    G_x1 = preprocess_obj.run_all(G_x1)
    G_y1 = preprocess_obj.run_all(G_y1)
    temp1 = orientation_field(G_x1, G_y1, 5)
    G_x2 = convolve2D(image2, kernel_x,padding=0)
    G_y2 = convolve2D(image2, kernel_y,padding=0)
    G_x2 = preprocess_obj.run_all(G_x2)
    G_y2 = preprocess_obj.run_all(G_y2)
    temp2 = orientation_field(G_x2, G_y2, 5)
    
    minutae_set1 = minutiae_extract(preprocessed_image1,temp1)
    minutae_set2 = minutiae_extract(preprocessed_image2,temp2)

    return minutae_set1,minutae_set2


# In[16]:


def open_database(path):
    files = os.listdir(path)
    images = []
    for current_file in files:
        temp_path = path+"/"+current_file
        im = Image.open(temp_path)
        images.append((current_file,np.array(im)))
    return images


# In[17]:


def generate_dataset(path,train_size = 16, test_size= 4):
    images = open_database(path)
    matching_prints= {}
    diff_fingerprints = 0
    for name,image in images:
        spl = name.split("_")
        if spl[0] in matching_prints:
            matching_prints[spl[0]].append((name,image))
        else:
            matching_prints[spl[0]] =[(name,image)]
            diff_fingerprints+=1
    train_prints_to_choose = random.sample(range(len(images)),train_size)
    test_prints_to_choose = random.sample(range(len(images)),test_size)
    
    test_set = []
    train_set = []
    for index,element in enumerate(images):
        if index in test_prints_to_choose:
            test_set.append(element)
        if index in train_prints_to_choose:
            train_set.append(element)
    return train_set,test_set
        


# In[18]:


def run_matching_algorithm(test_set,option):
    results = []
    for i in range(len(test_set)-1):
        s = test_set[i][0].split("_")[0]
        image1 = test_set[i][1]
        for j in range(i+1,len(test_set)):
            print(test_set[i][0],test_set[j][0])
            s2 = test_set[j][0].split("_")[0]
            image2= test_set[j][1]
            if option == 1:
                result = option1(image1,image2)
            else:
                result = option2(image1,image2)
            
            results.append((s==s2,result))
    return results
                


# In[19]:


def generate_metrics(results):
    acc = 0
    frr = 0
    far = 0
    #r[0] is gold, r[1] is mine
    for r in results:
        if r[0] == False and r[1] == True:
            far+=1
        if r[1] == True and r[0] == False:
            frr+=1
        if r[0] == r[1]:
            acc+=1
    acc/=len(results)
    frr/=len(results)
    far/=len(results)
    
    print("Accuracy:",acc*100,"%")
    print("FAR:",far*100,"%")
    print("FRR:",frr*100,"%")
    


# In[29]:


def engine(path="./DB1",training_size=10,test_size=8,option="1"):
    training_set,test_set = generate_dataset(path,training_size,test_size)
    training_minutiae_set = cache_training_set(training_set)
    if option == "1":
        results= cached_option_1(training_minutiae_set,test_set)
    else:
        results= cached_option_2(training_minutiae_set,test_set)
    generate_metrics(results)


# In[ ]:


# engine("./DB1",15,3)


# In[ ]:


if len(sys.argv) < 5:
    print("Usage: python assignment1.py <Path to Dataset> <Training Set Size> <Test Set Size> <1|2>")
else:
    path = sys.argv[1]
    tr_s_size = int(sys.argv[2])
    te_s_size = int(sys.argv[3])
    option = sys.argv[4]
    
    engine(path,tr_s_size,te_s_size,option)


# In[88]:


# filename1,im1 = open_database("DB1")[0]
# filename2,im2 = open_database("DB1")[2]
# minutiae_set1,minutiae_set2 = option2(im1,im2)


# In[ ]:


# def fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold):
#     #     Converting degrees to radian
#     theta = math.radians(theta) 
#     sin_theta = math.sin(theta)
#     cos_theta = math.cos(theta)

#     num = 0

#     for i in range(len(mx)):
#         px = s*(mx[i]*cos_theta - my[i]*sin_theta) - qx[i] + tx[i]
#         py = s*(mx[i]*sin_theta + my[i]*cos_theta) - qy[i] + ty[i]

#         if(math.sqrt(px*px + py*py)< thold):
#             num = num + 1
    
#     return num

# def convert_list_to_integer(lst):
#     sum = 0
#     base = 1
#     for i in range(len(lst)):
#         sum = sum + base*lst[i]
#         base = base*2
        
#     return sum

# def get_value_from_chromosome(single_chromo):
#     #     first 5 bits represents scale
#     s_list = single_chromo[:5]
# #     Next 6 bits represents angle
#     theta_list = single_chromo[5:11]
# #     Next 8 bits represents movement in x axis
#     tx_list = single_chromo[11:19]
# #     Next 8 bits represents movement in y axis
#     ty_list = single_chromo[19:27]
    
#     s = convert_list_to_integer(s_list)*0.01 + 0.9
#     theta = convert_list_to_integer(theta_list) - 30
#     tx = convert_list_to_integer(tx_list)-128
#     ty = convert_list_to_integer(ty_list)-128
    
#     return s, theta, tx, ty

# def Random_population_generation(n_pop):
#     #     n_pop is the total number of random population needed generally n_pop = total number of minutiae of query set.
#     pop = [randint(0, 2, 27).tolist() for _ in range(n_pop)]
#     return pop

# # tournament selection
# def selection(pop, scores, k=3):
#     # first random selection
#     selection_ix = randint(len(pop))
#     for ix in randint(0, len(pop), k-1):
#         # check if better (e.g. perform a tournament)
#         if scores[ix] < scores[selection_ix]:
#             selection_ix = ix
#     return pop[selection_ix]

# def Cross_over(p1, p2, r_cross):
#     # children are copies of parents by default
#     c1, c2 = p1.copy(), p2.copy()
#     # check for recombination
#     if rand() < r_cross:
#         # select crossover point that is not on the end of the string
#         pt = randint(1, len(p1)-2)
#         # perform crossover
#         c1 = p1[:pt] + p2[pt:]
#         c2 = p2[:pt] + p1[pt:]
#     return [c1, c2]

# def Mutation(bitstring, r_mut):
#     for i in range(len(bitstring)):
#         # check for a mutation
#         if rand() < r_mut:
#             # flip the bit
#             bitstring[i] = 1 - bitstring[i]

# def genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy):

#     # mx, my, tx, ty are list of numbers. mx, my are for template and qx, qy are for query
    
#     # initial population of random bitstring
#     pop = [randint(0, 2, 27).tolist() for _ in range(n_pop)]
    
#     # keep track of best solution
#     s, theta, tx, ty = get_value_from_chromosome(pop[0])
#     best, best_eval = 0, fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold)
    
#     # enumerate generations
#     for gen in range(n_iter):
        
#         # evaluate all candidates in the population
#         scores = []
#         for i in range(len(pop)):
#             s, theta, tx, ty = get_value_from_chromosome(pop[i])
#             scores.append(fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold))
        
#         # check for new best solution
#         for i in range(n_pop):
#             if scores[i] > best_eval:
#                 best, best_eval = pop[i], scores[i]
#                 print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
                
#         # select parents
#         selected = [selection(pop, scores) for _ in range(n_pop)]
        
#         # create the next generation
#         children = list()
#         for i in range(0, n_pop, 2):
#             # get selected parents in pairs
#             p1, p2 = selected[i], selected[i+1]
#             # crossover and mutation
#             for c in crossover(p1, p2, r_cross):
#                 # mutation
#                 mutation(c, r_mut)
#                 # store for next generation
#                 children.append(c)
#         # replace population
#         pop = children
        
#     return [best, best_eval]

# # define the total iterations
# n_iter = 100

# # define the population size. This varries as the number of miniuate of images varries
# n_pop = 100

# # crossover rate
# r_cross = 0.9

# # mutation rate
# r_mut = 1.0 / (float(27) * 2)

# thold = 1

# # perform the genetic algorithm search
# best, score = genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold)
# if(score>final_thresh):
#     print("Matched")
# else:
#     print("Not Matched")

# print('Done!')
# decoded = decode(bounds, n_bits, best)
# print('f(%s) = %f' % (decoded, score))

