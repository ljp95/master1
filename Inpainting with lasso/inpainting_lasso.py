
""" Created by ljp95 and Elodie Souksava on 05.26.2019 """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb

import time
from sklearn import linear_model

""" About USP data """

def load_usps(filename):
    """ Load USPS data """
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    """ Display a data/vector of USPS as an image """
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def extract_data(classe1,classe2,datax,datay):
    """ Extract and process input and output data from chosen classes """
    indices = np.logical_or(datay == classe1, datay == classe2)
    extractx = np.copy(datax[indices])
    extracty = np.copy(datay[indices])
    indices_plus = extracty == classe1 
    indices_minus = np.logical_not(indices_plus)
    extracty[indices_plus] = 1
    extracty[indices_minus] = -1
    return extractx,extracty

def infos_usp(name,clf,w,trainx,trainy,testx,testy):
    """ Create a figure with name 
    Give train and test scores, number of zeros and norm from the weights vector """
    # figure
    x = range(len(w))
    plt.figure()
    plt.title(name)
    plt.scatter(x,w,s=2)
    plt.legend()
    plt.xlabel('i-th coeff')
    plt.ylabel('wi values')
    # print informations
    print(name)
    print('score train : {} \nscore test : {}'.format(clf.score(trainx,trainy),clf.score(testx,testy)))
    print('number of zeros : ',sum(w == 0))
    print('norm of w : ',np.sqrt(np.sum(w**2)))
    return

####################### APPROXIMATION WITH LASSO PART ########################
    
# to recognize dead pixels
DEAD = -100

def read_image(file):
    """ read an image and normalize it """
    image = plt.imread(file)/255
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in range(3):
        mean[i] = np.mean(image[:,:,i])
        std[i] = np.std(image[:,:,i])
    image2 = (image-mean)/std
    return image2,mean,std
def read_image(file):
    """ read an image and normalize it """
    image = plt.imread(file)/255
    return image,0,0
def show_image(image,mean,std):
    ''' display image by un-normalize it '''
    indices = image==DEAD
    image2 = image.copy()
    image2[indices] = 0
    plt.figure()
    plt.imshow(image2)
    return 

def zoom(image,mean,std,line,col,height,length):
    """ Display a zoom centered on a point """
    show_image(image[line-height:line+height,col-length:col+length,:],mean,std)
    return

def out(line,col,h,image):
    """ Checking if a patch centered on (i,j) is out of range """
    n,m = len(image),len(image[0])
    if((n-h)<line<h and (m-h-1)<col<h):
        print('Out of range!')
        return True
    return False

def get_patch(line,col,h,image):
    """ Get the patch centered on (i,j) of size 2*h+1 """
    if not out(line,col,h,image):
        return image[(line-h):(line+h+1),(col-h):(col+h+1)]
    return None

def patch_to_vector(patch):
    """ Transform a patch to a vector """
    return patch.reshape(-1)
    
def vector_to_patch(vector,h):
    """ Transform a vector to a patch of size 2*h+1 """
    return vector.reshape((2*h+1,2*h+1,3))

def construct_vectors(image,h):
    """ Return all patches, non-noisy patches, noisy patches directly as vectors """
    #init parameters
    n,m = len(image),len(image[0])
    vectors = []
    complete_vectors = []
    incomplete_vectors = []
    i = h
    step = h
    # Getting all patches, transform and separate them
    while(i<(n-h)):
        j = h
        while(j<(m-h)):
            patch = get_patch(i,j,h,image)
            vector = patch_to_vector(patch)
            vectors.append(vector)
            if DEAD in patch:
                incomplete_vectors.append(vector)
            else:
                complete_vectors.append(vector)
            j += step
        i += step
    return np.array(vectors),np.array(complete_vectors),np.array(incomplete_vectors)

def predict(vector,trainx,alpha=0.0001,max_iter=50000):
    """ Return weights and approximation of the lasso """
    lasso = linear_model.Lasso(alpha=alpha,max_iter=max_iter,tol=0.00000001)
    indices = vector!= DEAD 
    lasso.fit(trainx.T[indices],vector[indices])
    w = lasso.coef_
    return w,lasso.predict(trainx.T)

def approximation(vector,trainx,h,mean,std,alpha=0.0001,max_iter=50000,noise = 0):
    """ Display an approximation of a patch (noisy or not) and return the weights """
    debut = time.time()
    w,approx_vector = predict(vector,trainx,alpha = alpha,max_iter = max_iter)
    print(time.time()-debut)
    approx_patch = vector_to_patch(approx_vector,h)
    if noise:
        add = 'denoised'
    else:
        add = 'approximated'
    plt.figure();show_image(approx_patch,mean,std);plt.title("patch {} alpha : {}".format(add,alpha))
    return w

def infos_inpainting(w,alpha):
    """ Create a figure with name 
    Give train and test scores, number of zeros and norm from the weights vector """
    x = range(len(w))
    plt.figure()
    plt.title('lasso alpha = {}'.format(alpha))
    plt.scatter(x,w,s=2)
    plt.xlabel('i-th coeff')
    plt.ylabel('w values')
    print('number of zeros : ',sum(w == 0))
    print('norm of wi : ',np.sqrt(np.sum(w**2)))
    return

def noise(image,prc):
    """ Return an image with aroung 'prc' of noise randomly distributed """
    n,m = len(image),len(image[0])
    indices_i = np.random.choice(np.arange(n),int(prc*n*m))
    indices_j = np.random.choice(np.arange(m),int(prc*n*m))
#    indices = tuple(zip(indices_i,indices_j))
    image2 = image.copy()
    image2[[indices_i,indices_j]] = DEAD
    return image2

def delete_rect(image,i,j,height,length):
    """ Return an image with a rectangle of dead pixels centered on (i,j) """
    if out(i,j,max(height,length),image):
        return image
    image2 = image.copy()
    image2[i-height:i+height+1,j-length:j+length+1,:] = DEAD
    return image2

############################### FILLING PART #################################

def update(image_noisy,line,col,h,trainx):
    """ update the image at (line,col) with a patch approximated by lasso """
    vector = patch_to_vector(get_patch(line,col,h,image_noisy))
    w,approx_vector = predict(vector,trainx)
    approx_patch = vector_to_patch(approx_vector,h)
#    trainx = np.concatenate([trainx,approx_vector.reshape(1,-1)])
    image_noisy[line-h:line+h+1,col-h:col+h+1,:] = approx_patch

def check_edge(l,c,image_noisy):
    """ Checking if a point(l,c) is from the contour of a missing part of the image """
    return np.sum(image_noisy[l-1:l+2,c-1:c+2,0] == DEAD)<9 and image_noisy[l,c,0]==DEAD

def simple_filling(image_noisy,mean,std,trainx,line,col,height,length,h):
    """ Filling in the following order : first pixel encountered from the edge """
    end = False
    image2 = image_noisy.copy()
    while not end:
        end = True
        # Scan all columns of each line until a component of the missing part's contour is found """
        for l in range(line-height,line+height+1):
            for c in range(col-length,col+length+1):
                if check_edge(l,c,image2):
                    end = False
                    update(image2,l,c,h,trainx)
                    break
            if end == False:
                break
        # display a zoom on the filling part
        zoom(image2,mean,std,line,col,height*2,length*2)
    return image2

def C(patch):
    """ Compute the confidence on a point """
    indices = patch[:,:,0]!= DEAD
    return sum(sum(indices)/(len(patch)*len(patch[0])))

def D(l,c,isoX,isoY,nX,nY):
    """ Compute data on a point """
    return np.abs(isoX[l,c]*nX + isoY[l,c]*nY)

def edge_and_normal(image_noisy,line,col,height,length):
    """ Search for edge with 8 directions and compute normals """
    edge = []
    # Scan all columns of each line until a component of the missing part's contour is found """
    for l in range(line-height,line+height+1):
        for c in range(col-length,col+length+1):
            if check_edge(l,c,image_noisy):
                edge.append([l,c])
    if len(edge) == 0:
        return None,None
    # as a result we have an edge with an order of appearance like this 
    """ 123
        4 5
        678 """
    # need to sort it for the normals like this
    """ 123
        8 4
        765 """
    edge = sort_edge(edge)
    # Compute normals to the edge
    normals = []
    for i in range(len(edge)-1):
        [l,c] = edge[i]
        [l2,c2] = edge[i+1]
        normals.append([l2-l,c2-c])
    return edge,normals

def sort_edge(edge):
    """ sort the points so as they follow each other in the list 
        edge with 8 directions
        going from 123   to 123
                   4 5      8 4
                   678      765 """
    # get a first point
    edge2 = [edge.pop(0)]
    n = len(edge)
    # looping until the new list of the contour is filled
    while(len(edge2)<n+1):
        [l,c] = edge2[-1]
        valeur = 3 # to favor a vertical or horizontal direction over a diagonal
        suivant = 0
        # looking for the next point, we move horizontaly and/or verticaly by one
        for i in range(len(edge)):
            [l2,c2] = edge[i]
            dy = np.abs(l2-l)
            dx = np.abs(c2-c)
            # to favor a horizontal or a vertical move over a diagonal move
            if dx<=1 and dy<=1 and dx+dy<valeur:
                valeur = dx+dy
                suivant = i
        edge2.append(edge.pop(suivant))
    edge2.append(edge2[0]) # adding the first point again to get a cycle
    return edge2

def isophote(image_noisy_rgb,image_noisy):
    """ Compute isophotes by computing and rotating by 90° the gradient
    Need the grey and color values of the image """
    n,m,d = image_noisy.shape
    isox = np.zeros((n,m,d))
    isoy = np.zeros((n,m,d))
    for i in range(d):
        isoy[:,:,i],isox[:,:,i] = np.gradient(image_noisy_rgb[:,:,i]) 
    isox,isoy = np.sum(isox,axis=2)/255,np.sum(isoy,axis=2)/255
    isox,isoy = -isoy,isox #rotation de 90°
    return isox,isoy

def data_and_confidence(image,h,edge,normal,isox,isoy):
    """ Compute all data and confidence of the edge """
    confidence = []
    data = []
    for i in range(len(edge)-1):
        [l,c] = edge[i]
        patch = image[l-h:l+h+1,c-h:c+h+1]
        confidence.append(C(patch))
        nx,ny = normal[i]
        data.append(D(l,c,isox,isoy,nx,ny))
    return np.array(data),np.array(confidence)

def better_filling(image_noisy_rgb,image_noisy,mean,std,trainx,line,col,height,length,h):
    ''' Filling by priorities which are computed by taking in account data and confidence on each points of the edge'''
    image2 = image_noisy.copy()
    #contour and normals
    edge,normal = edge_and_normal(image_noisy,line,col,height,length)
    #isophotes
    isox,isoy = isophote(image_noisy_rgb,image_noisy)
    end = False
    #looping as long there is dead pixels
    while(not end):
        #data and confidence
        data,confidence = data_and_confidence(image2,h,edge,normal,isox,isoy)
        #priorities
        priorities = confidence * data 
        
        [l,c] = edge[priorities.argmax()]
        update(image2,l,c,h,trainx)    
        zoom(image2,mean,std,line,col,height*2,length*2)
        
        #if there is still dead pixels or not
        edge,normal = edge_and_normal(image2,line,col,height,length)
        end = edge == None
    return image2
