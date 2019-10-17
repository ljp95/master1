from ex1 import *
import random
import time
import copy

def glouton_init(photos,numerosH_dispo,numerosV_dispo):
    ''' renvoit la premiere slide du glouton1 (simple). Slide choisie au hasard
    Gere les cas sans photosV ou photosH
    entrees : photos, numeros des photosH et photosV disponibles
    sortie : une slide
    '''
    if len(numerosV_dispo)>=2:
        if len(numerosH_dispo) != 0:
            if(random.random()<0.5):
                slide = [random.choice(numerosH_dispo)]
            else:
                slide = random.sample(numerosV_dispo,2)
        else:
            slide = random.sample(numerosV_dispo,2)
    else:
        slide = [random.choice(numerosH_dispo)]
    return slide

def glouton_init2(photos,numerosH_dispo,numerosV_dispo):
    ''' renvoit la premiere slide du glouton2 (plus rapide). 
    Renvoit toujours une slide de photoH si possible, photosV sinon
    entrees : photos, numeros des photosH et photosV disponibles
    sortie : une slide
    '''
    if(len(numerosH_dispo)):
        slide = [random.choice(numerosH_dispo)]
    else:
        slide = random.sample(numerosV_dispo,2)
    return slide

def best_photo(photos,numeros_dispo,tags1,photo1=None):
    ''' recherche la photo qui maximise la transition a une slide parmi toutes les photos dispo
    Si photo1 = None on cherche une photo H ou V sinon on recherche une photo V pour completer photo1
    entrees : photos, numeros disponibles, le set de tags de la slide precedente, photo1 si on cherche a completer
    sorties : une photo et le score de transition en prenant cette derniere
    '''
    if photo1 == None:
        score_actuel = -1
        for i in numeros_dispo:
            tags2 = photos[i][2]
            score = score_transition(tags1,tags2)
            if(score>score_actuel):
                score_actuel = score
                photo1 = i
        return photo1,score_actuel
    else:
        #seule difference avec au dessus : pour calculer la transition on prend tags([photo1,photo])
        score_actuel = -1
        photo2 = None
        for i in numeros_dispo:
            tags2 = tags([photo1,i],photos)
            score = score_transition(tags1,tags2)
            if(score>score_actuel):
                score_actuel = score
                photo2 = i
        return photo2,score_actuel
    
def best_slide1(photos,numerosH_dispo,numerosV_dispo,tags1):
    ''' renvoit une slide proche ou egale a la maximisation de la transition 
    recherche la photo qui maximise la transition.
    si photo horizontale : maximisation
    si photo verticale, recherche une deuxieme photo verticale qui maximise la transition
    mais pas de nouvelle comparaison avec le score de la photo horizontale
    entrees : photos, numeros de photosH et photosV dispo, set de tags de la slide precedente
    sorties : deux photos H ou V ou de forme None
    '''
    photoH,scoreH = best_photo(photos,numerosH_dispo,tags1)
    photoV,scoreV = best_photo(photos,numerosV_dispo,tags1)
    
    if scoreH>=scoreV:
        return photoH,None
    
    #cas d'une meilleur photo verticale
    else:
        #retirer la photo verticale des possibilites
        numerosV_dispo = list(set(numerosV_dispo)-set([photoV]))
        photoV2,tmp = best_photo(photos,numerosV_dispo,tags1,photoV)
    if photoV2 == None:
        return photoH,None
    return photoV,photoV2

def best_slideV(photos,numerosV_dispo,tags1):
    ''' renvoit les deux photos verticales qui maximisent separement la transition 
    on utilise alors qu'une seule boucle au lieu de reparcourir les photos verticales par rapport au glouton1
    entrees : photos,numerosV disponibles et le set de tag de la slide precedente
    sorties : deux photosV
    '''
    score1,score2 = -1,-1
    photo1,photo2 = None,None
    for i in numerosV_dispo:
        tags2 = photos[i][2]
        score = score_transition(tags1,tags2)
        if score>=score1:
            score1,score2 = score,score1
            photo1,photo2 = i,photo1
        else: 
            if score>score2:
                score2 = score
                photo2 = i
    return photo1,photo2
    
def best_slide2(photos,numerosH_dispo,numerosV_dispo,tags1):
    ''' renvoit une slide proche ou egale a la maximisation de la transition 
    on renvoit les photos horizontales maximiseurs d'abord
    s'il n'y en a plus, recherche de deux photos verticales qui maximisent separement la transition 
    entrees : photos,numeros H et V disponibles, set de tag de la slide precedente
    sorties : deux photos H ou V ou de forme None
    '''
    if(numerosH_dispo):
        photoH,scoreH = best_photo(photos,numerosH_dispo,tags1)
        return photoH,None
    else:
        photo1,photo2 = best_slideV(photos,numerosV_dispo,tags1)
    #si qu'une photo verticale i.e photo2=None, la slide n'est pas possible donc on renvoit photo1=None
    if photo2 == None:
        return None,None
    return photo1,photo2
    
def maj(presentation,photos,numerosH_dispo,numerosV_dispo,slide):
    ''' met a jour la presentation, les numeros de photos disponibles selon la nouvelle slide
    entrees : presentation, numeros H et V disponibles et la nouvelle slide
    sortie : set de tags de la nouvelle slide qu'on comparera aux nouvelles slides potentielles
    '''
    presentation.append(slide)
    if len(slide) == 2:
        numerosV_dispo.remove(slide[0])
        numerosV_dispo.remove(slide[1])
    else:
        numerosH_dispo.remove(slide[0])
    return tags(slide,photos)

def algo_glouton(instance,temps_limite = 100000,glouton=2):
    ''' renvoit une presentation glouton selon la fonction maximiseur choisie 
    entrees : instance, un temps limite, le numero de glouton voulu
    sortie : une presentation
    '''
    #initialisation des parametres
    debut = time.time()
    photos,numerosH,numerosV = instance
    numerosH_dispo,numerosV_dispo = copy.deepcopy(numerosH),copy.deepcopy(numerosV)
    presentation = []
    
    #premiere slide et fonction maximiseur selon le glouton choisi
    if(glouton == 2):
        maximiseur = best_slide2
        slide1 = glouton_init2(photos,numerosH_dispo,numerosV_dispo)
    else: 
        maximiseur = best_slide1
        slide1 = glouton_init(photos,numerosH_dispo,numerosV_dispo)
    
    #mise a jour de la presentation, des numeros dispo et du set de tags actuel
    tags1 = maj(presentation,photos,numerosH_dispo,numerosV_dispo,slide1)
    
    #sortie du while si temps limite depasse ou pas de slide convenable <=> photo1 == None
    while(time.time()-debut < temps_limite):
        #recherche de la slide suivante ~maximiseur
        photo1,photo2 = maximiseur(photos,numerosH_dispo,numerosV_dispo,tags1)
        #break si pas de slide disponible 
        if photo1 == None:
            break
        #met les photos sous forme d'une slide/liste
        else:
            if photo2 != None:
                slide2 = [photo1,photo2]
            else:
                slide2 = [photo1]
        #mise a jour de la presentation, des numeros dispo et du set de tags actuel
        tags1 = maj(presentation,photos,numerosH_dispo,numerosV_dispo,slide2)
        if(len(presentation)%100 == 0):
            print("{} slides en {} secs".format(len(presentation),time.time()-debut))
    return presentation


