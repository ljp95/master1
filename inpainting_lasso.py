import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import time
from sklearn import linear_model

##################### PREAMBULE #####################

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def extract_data(classe1,classe2,datax,datay):
    ''' extrait les donnees issues des classes en parametres '''
    indices = np.logical_or(datay == classe1, datay == classe2)
    extractx = np.copy(datax[indices])
    extracty = np.copy(datay[indices])
    indices_plus = extracty == classe1 
    indices_moins = np.logical_not(indices_plus)
    extracty[indices_plus] = 1
    extracty[indices_moins] = -1
    return extractx,extracty

def infos_usp(nom,clf,w,trainx,trainy,testx,testy):
    ''' donne scores train et test, norme et nombre de zeros du vecteur de poids '''
    x = range(len(w))
    plt.figure()
    plt.title(nom)
    plt.scatter(x,w,s=2)
    plt.legend()
    plt.xlabel('i-ème coeff')
    plt.ylabel('valeur')
    print(nom)
    print('score train : {} \nscore test : {}'.format(clf.score(trainx,trainy),clf.score(testx,testy)))
    print('nombre de zeros : ',sum(w == 0))
    print('norme de w : ',np.sqrt(np.sum(w**2)))
    return

##################### LASSO ET INPAINTING #####################
    
#pixels manquants
DEAD = -100

def read_image(file):
    ''' image avec normalisation des donnees '''
    image = plt.imread(file)/255
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in range(3):
        mean[i] = np.mean(image[:,:,i])
        std[i] = np.std(image[:,:,i])
    image2 = (image-mean)/std
    return image2,mean,std

def show_image(image,mean,std):
    ''' affichage de l'image '''
    indices = image==DEAD
    image2 = image.copy()*std+mean
    image2[indices] = 0
    plt.figure()
    plt.imshow(image2)
    return 

def zoom(image,mean,std,line,col,height,length):
    ''' affichage zoomé en un point '''
    show_image(image[line-height:line+height,col-length:col+length,:],mean,std)
    return

def out(line,col,h,image):
    ''' verifie si le patch centre en (i,j) est possible '''
    n,m = len(image),len(image[0])
    if (n-h)<line<h and (m-h-1)<col<h:
        print('Hors-limite!')
        return True
    return False

def get_patch(line,col,h,image):
    ''' recupere le patch centre en (i,j) de taille 2*h+1 '''
    if not out(line,col,h,image):
        return image[(line-h):(line+h+1),(col-h):(col+h+1)]
    return None

def patch_to_vector(patch):
    ''' conversion patch en vecteur '''
    return patch.reshape(-1)
    
def vector_to_patch(vector,h):
    ''' conversion vecteur en patch de taille 2*h+1 '''
    return vector.reshape((2*h+1,2*h+1,3))

def construct_vectors(image,h):
    ''' renvoit tous les patchs, patchs non bruites, patch bruites sous forme de vecteurs directement '''
    #init parametres
    n,m = len(image),len(image[0])
    vectors = []
    complete_vectors = []
    incomplete_vectors = []
    i = h
    step = h
    #boucle 
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

def predict(vector,trainx,alpha=0.00001,max_iter=10000):
    ''' renvoit les poids et l'approximation du lasso '''
    lasso = linear_model.Lasso(alpha=alpha,max_iter=max_iter,tol=0.00000001)
    indices = vector!= DEAD 
    lasso.fit(trainx.T[indices],vector[indices])
    w = lasso.coef_
    return w,lasso.predict(trainx.T)

def approximation(vector,trainx,h,mean,std,alpha=0.00001,max_iter=10000,noise = 0):
    debut = time.time()
    w,approx_vector = predict(vector,trainx,alpha = alpha,max_iter = max_iter)
    print(time.time()-debut)
    approx_patch = vector_to_patch(approx_vector,h)
    if noise:
        add = 'debruitee'
    else:
        add = 'approxime'
    plt.figure();show_image(approx_patch,mean,std);plt.title("patch {} alpha : {}".format(add,alpha))
    return w

def infos_inpainting(w,alpha):
    ''' donne scores train et test, norme et nombre de zeros du vecteur de poids '''
    x = range(len(w))
    plt.figure()
    plt.title('lasso alpha = {}'.format(alpha))
    plt.scatter(x,w,s=2)
    plt.xlabel('i-ème coeff')
    plt.ylabel('valeur')
    print('nombre de zeros : ',sum(w == 0))
    print('norme de w : ',np.sqrt(np.sum(w**2)))
    return

def noise(image,prc):
    ''' renvoit une image avec environ 'prc' de bruit '''
    n,m = len(image),len(image[0])
    indices_i = np.random.choice(np.arange(n),int(prc*n*m))
    indices_j = np.random.choice(np.arange(m),int(prc*n*m))
#    indices = tuple(zip(indices_i,indices_j))
    image2 = image.copy()
    image2[[indices_i,indices_j]] = DEAD
    return image2

def delete_rect(image,i,j,height,length):
    ''' renvoit une image bruite d'un rectangle de centre (i,j) '''
    if out(i,j,max(height,length),image):
        return image
    image2 = image.copy()
    image2[i-height:i+height+1,j-length:j+length+1,:] = DEAD
    return image2
