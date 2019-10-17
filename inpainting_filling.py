from lasso import *

####################### REMPLISSAGE ##############################

def maj(image_noisy,line,col,h,trainx):
    ''' mise a jour de l'image par un patch approxime par le lasso '''
    vector = patch_to_vector(get_patch(line,col,h,image_noisy))
    w,approx_vector = predict(vector,trainx)
    approx_patch = vector_to_patch(approx_vector,h)
#    trainx = np.concatenate([trainx,approx_vector.reshape(1,-1)])
    image_noisy[line-h:line+h+1,col-h:col+h+1,:] = approx_patch
    return 

def check_edge(l,c,image_noisy):
    ''' verifie si un point est issu d'un contour ou non '''
    return np.sum(image_noisy[l-1:l+2,c-1:c+2,0] == DEAD)<9 and image_noisy[l,c,0]==DEAD

def simple_fill(image_noisy,mean,std,trainx,line,col,height,length,h):
    ''' remplissage simple : premier pixel du contour rencontre '''
    fin = False
    image2 = image_noisy.copy()
    while not fin:
        fin = True
        #parcourt toutes les colonnes de chaque ligne jusqu'à trouver un élément du contour
        for l in range(line-height,line+height+1):
            for c in range(col-length,col+length+1):
                if check_edge(l,c,image2):
                    fin = False
                    maj(image2,l,c,h,trainx)
                    break
            if fin == False:
                break
        zoom(image2,mean,std,line,col,height*2,length*2)
    return image2

def C(patch):
    ''' calcul de la confiance en un point '''
    indices = patch[:,:,0]!= DEAD
    return sum(sum(indices)/(len(patch)*len(patch[0])))

def D(l,c,isoX,isoY,nX,nY):
    ''' calcul de data en un point '''
    return np.abs(isoX[l,c]*nX + isoY[l,c]*nY)

def edge_and_normal(image_noisy,line,col,height,length):
    ''' recherche du contour et calcul des normales '''
    edge = []
    #parcourt toutes les colonnes de chaque ligne jusqu'à trouver un élément du contour
    for l in range(line-height,line+height+1):
        for c in range(col-length,col+length+1):
            if check_edge(l,c,image_noisy):
                edge.append([l,c])
    if len(edge) == 0:
        return None,None
    #contour pour l'instant dans l'ordre d'apparition 
    # -> a trier de sorte que les points se suivent dans la liste pour les normales
    edge = sort_edge(edge)
    #calcul des normales au contour
    normal = []
    for i in range(len(edge)-1):
        [l,c] = edge[i]
        [l2,c2] = edge[i+1]
        normal.append([l2-l,c2-c])
    return edge,normal

def sort_edge(edge):
    ''' trie les points de contours de sorte qu'ils se suivent dans la liste '''
    edge2 = [edge.pop(0)]
    n = len(edge)
    while(len(edge2)<n+1):
        [l,c] = edge2[-1]
        valeur = 3 # pour preferer une fleche horizontale/verticale au lieu d'une diagonale
        suivant = 0
        #recherche du point suivant : deplacement unitaire en x et/ou y
        for i in range(len(edge)):
            [l2,c2] = edge[i]
            dy = np.abs(l2-l)
            dx = np.abs(c2-c)
            if dx<=1 and dy<=1 and dx+dy<valeur:
                valeur = dx+dy
                suivant = i
        edge2.append(edge.pop(suivant))
    edge2.append(edge2[0]) #boucler le contour pour les normales
    return edge2

def isophote(image_noisy_rgb,image_noisy):
    ''' calcul des isophotes en tout point : 
        calcul puis rotation du gradient de 90° 
        besoin des valeurs gray ou rgb de base de l'image '''
    n,m,d = image_noisy.shape
    isox = np.zeros((n,m,d))
    isoy = np.zeros((n,m,d))
    for i in range(d):
        isoy[:,:,i],isox[:,:,i] = np.gradient(image_noisy_rgb[:,:,i]) #np.gradient retourne y d'aedge
    isox,isoy = np.sum(isox,axis=2)/255,np.sum(isoy,axis=2)/255
    isox,isoy = -isoy,isox #rotation de 90°
    return isox,isoy

def data_and_confiance(image,h,edge,normal,isox,isoy):
    confiance = []
    data = []
    for i in range(len(edge)-1):
        [l,c] = edge[i]
        patch = image[l-h:l+h+1,c-h:c+h+1]
        confiance.append(C(patch))
        nx,ny = normal[i]
        data.append(D(l,c,isox,isoy,nx,ny))
    return np.array(data),np.array(confiance)

def better_fill(image_noisy_rgb,image_noisy,mean,std,trainx,line,col,height,length,h):
    ''' remplissage suivant l'article :
        calcul des confiances, data (isophotes*normal) pour calculer les priorites '''
    image2 = image_noisy.copy()
    #contour et normales
    edge,normal = edge_and_normal(image_noisy,line,col,height,length)
    #isophotes
    isox,isoy = isophote(image_noisy_rgb,image_noisy)
    fin = False
    #tant qu'il reste des pixels morts
    while(not fin):
        #data et confiance
        data,confiance = data_and_confiance(image2,h,edge,normal,isox,isoy)
        #priorité
        priorite = confiance * data 
        
        [l,c] = edge[priorite.argmax()]
        maj(image2,l,c,h,trainx)    
        zoom(image2,mean,std,line,col,height*2,length*2)
        
        #si il reste des points morts ou non 
        edge,normal = edge_and_normal(image2,line,col,height,length)
        fin = edge == None
    return image2
