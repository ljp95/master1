from lasso import *
from fill import *

##################### PREAMBULE #####################
##choix des classes
#classe1 = 1
#classe2 = 7
##extraction donnees selon classes
#datax,datay = load_usps('USPS_train.txt')
#trainx, trainy = extract_data(classe1,classe2,datax,datay)
#datax,datay = load_usps('USPS_test.txt')
#testx,testy = extract_data(classe1,classe2,datax,datay)
#
#
###### Regression logistique
#reg = linear_model.LogisticRegression()
#reg.fit(trainx,trainy)
#nom = 'regression logistique {} vs {}'.format(classe1,classe2)
#infos_usp(nom,reg,reg.coef_[0],trainx,trainy,testx,testy)
#
#
###### Ridge regression
#alpha = 10
#ridge = linear_model.Ridge(alpha = alpha)
#ridge.fit(trainx,trainy)
#nom = 'regression ridge {} vs {}'.format(classe1,classe2)
#infos_usp(nom,ridge,ridge.coef_,trainx,trainy,testx,testy)
#
#
###### Lasso
#alpha = 0.001
#lasso = linear_model.Lasso(alpha = alpha)
#lasso.fit(trainx,trainy)
#nom = 'lasso {} vs {}'.format(classe1,classe2)
#infos_usp(nom,lasso,lasso.coef_,trainx,trainy,testx,testy)


##################### LASSO ET INPAINTING #####################

h = 20

#image
file = 'ocean.jpg'
image,mean,std = read_image(file)
plt.figure();show_image(image,mean,std);plt.title("image originale")


#construction des patchs
vectors,complete_vectors,incomplete_vectors = construct_vectors(image,h)
trainx = complete_vectors


##### Test sur un patch
line,col = 207,207 #horizon
line,col = 95,395 #nuage
line,col = 285,285 #ocean


#patch original
patch = get_patch(line,col,h,image)
plt.figure();show_image(patch,mean,std);plt.title("patch original")
vector = patch_to_vector(patch)


#Test patch non bruite
alpha = 0.01
max_iter = 1000
w = approximation(vector,trainx,h,mean,std)
infos_inpainting(w,alpha)


#patch bruite 
patch_noisy = noise(patch,0.5)
vector_noisy = patch_to_vector(patch_noisy)
plt.figure();show_image(patch_noisy,mean,std);plt.title("patch bruite")


#test patch bruite
w = approximation(vector_noisy,trainx,h,mean,std,noise = 1)
infos_inpainting(w,alpha)


#Ajout de bruit sur toute l'image
prc = 0.5
image_noisy = noise(image,prc)
show_image(image_noisy,mean,std)


#Ajout d'un rectangle de bruit 
height = 20
length = 20
image_noisy = delete_rect(image,line,col,height,length)
show_image(image_noisy,mean,std)
zoom(image_noisy,mean,std,line,col,height*2,length*2)


#simple fill
debut = time.time()
vectors,trainx,incomplete_patches = construct_vectors(image,h)
image2 = simple_fill(image_noisy,mean,std,trainx,line,col,height,length,h) 
print("Approximation en {}secs".format(time.time()-debut))  
show_image(image2,mean,std) 


#better fill
image_noisy_rgb = plt.imread(file)*1.0
debut = time.time()
vectors,trainx,incomplete_patches = construct_vectors(image,h)
image2 = better_fill(image_noisy_rgb,image_noisy,mean,std,trainx,line,col,height,length,h) 
print("Approximation en {}secs".format(time.time()-debut))  
show_image(image2,mean,std) 
        
        
                
                