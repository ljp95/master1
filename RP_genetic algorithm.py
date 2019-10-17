from ex1 import *
import numpy as np
import random
import time

############################## 4.1 DESCENTE STOCHASTIQUE
def score_voisinage(pres,photos,ind1,ind2):
    ''' renvoit la somme des scores de transition des slides d'une presentation affectees par indice1 et indice2 
    entrees : presentation, les deux indices
    sortie : score de transition des slides autour des deux indices
    '''
    score = 0
    for i in range(2):
        #i=0 : cas du score de transition entre indice-1 et indice
        #i=1 : cas du score de transition entre indice et indice+1
        score += score_transition(tags(pres[ind1-1+i],photos),tags(pres[ind1+i],photos)) 
        score += score_transition(tags(pres[ind2-1+i],photos),tags(pres[ind2+i],photos))
    return score

def algo_descente(pres,photos,max_iter = None):
    ''' descente stochastique par tirage au sort entre echanger deux slides 
    et permuter des photos verticales entre deux slides 
    entrees : presentation et un nombre d'iteration
    sortie : presentation apres la descente stochastique
    '''
    #initialisation des parametres
    if max_iter == None:
        max_iter = len(pres)//2
    score = 0
    score_desc = 0
    numeros_slide = list(range(1,len(pres)-1))
    numeros_slideV = [i for i in range(1,len(pres)-1) if len(pres[i]) == 2]
    
    #si pas de photosV, on ne fait que des echanges de slide sinon 50/50
    if numeros_slideV != []:
        p = 0.5
    else:
        p = 1    
    for i in range(max_iter):
        r = random.random()
        if(r<p):
            '''calcul des scores de voisinage sans et avec l'echange de 2 slides
            la presentation est alors modifiee'''
            
            [ind1,ind2] = random.sample(numeros_slide,2)
            score = score_voisinage(pres,photos,ind1,ind2)
            pres[ind1],pres[ind2] = pres[ind2],pres[ind1]
            score_desc = score_voisinage(pres,photos,ind1,ind2)

            '''refaire l'echange si le score de voisinage de la descente est moins bonne
            sinon on a alors augmente le score donc
            mettre a jour les numeros des slides verticales 
            dans le cas ou on a fait un echange entre une slide horizontale et une de verticales 
            par besoin de tracker les numeros de slides verticales pour le second cas'''
            
            if(score_desc<score):
                pres[ind1],pres[ind2] = pres[ind2],pres[ind1]
            
            else:
                if(len(pres[ind1]) == 2 and len(pres[ind2]) == 1):
                    numeros_slideV.append(ind1)
                    numeros_slideV.remove(ind2)
                else:
                    if(len(pres[ind1]) == 1 and len(pres[ind2]) == 2):
                        numeros_slideV.append(ind2)
                        numeros_slideV.remove(ind1)
        else:
            #calcul des scores de voisinage sans et avec l'echange de photos verticales de deux slides
            [ind1,ind2] = random.sample(numeros_slideV,2)
            score = score_voisinage(pres,photos,ind1,ind2)
            [ind11,ind22] = random.sample([0,0,1,1],2)
            pres[ind1][ind11],pres[ind2][ind22] = pres[ind2][ind22],pres[ind1][ind11]
            score_desc = score_voisinage(pres,photos,ind1,ind2)
            
            #refaire l'echange si le score de voisinage de la descente est moins bonne
            if(score_desc<score):
                pres[ind1][ind11],pres[ind2][ind22] = pres[ind2][ind22],pres[ind1][ind11]
    return pres

######################################## 4.2 ALGORITHME GENETIQUE
def generer_population(taille, instance):
    ''' Generation d'une liste de presentations simples i.e de slides H puis V dans l'ordre d'arrivee (Ex1) a partir de shuffles de l'instance
    On separe chaque individu en deux. Une partie H et une partie V
    entrees : taille de la population, instance
    sorties : deux listes de population. Une pour la partie H une autre pour V
    '''
    #initialisation des parametres
    photos,numerosH,numerosV = instance    
    populationH = []
    populationV = []
    numerosH_random,numerosV_random = numerosH.copy(), numerosV.copy()
    
    for i in range(taille):
        np.random.shuffle(numerosH_random)
        np.random.shuffle(numerosV_random)
        populationH.append(presentation_simple([photos,numerosH_random,[]]))
        populationV.append(presentation_simple([photos,[],numerosV_random]))
    return populationH,populationV

def selection(population,probas):
    ''' renvoit deux individus de la population selon leur probabilite d'etre choisi 
    entrees : population et une liste de probas de chaque individu
    sorties : deux individus
    '''
    ind1,ind2 = np.random.choice(range(len(population)),2,replace=False,p=probas)
    return population[ind1],population[ind2]

def rangement_par_qualite(n,p):
    ''' renvoit les probabilites de chaque individu en fonction de leur rang : p((1-p)^rang) 
    entrees : taille de la population, hyperparametre p
    sortie : liste de probas p((1-p)^rang directement ordonnees de maniere decroissante
    '''
    probas = p*np.power(1-p,np.arange(n))
    probas /= probas.sum()
    return probas

def croisement(x,y):
    ''' renvoit deux fils issus de x et y
    croisement en un point seulement pour les photos horizontales
    pas de solution proposee pour le croisement des photos horizontales
    entrees : parents x et y
    sorties : deux fils z1 et z2
    '''
    #initialisation des parametres
    pos = random.randint(1,len(x)-1)
    z1 = x[pos:]
    z2 = y[pos:]
    
    numeros1 = [slide[0] for slide in z1]
    numeros2 = [slide[0] for slide in z2]
    
    for slide in y:
        if slide[0] not in numeros1:
            z1.append(slide)
    for slide in x:
        if slide[0] not in numeros2:
            z2.append(slide)
    return z1,z2

def next_gen(taille, photos, population, s, res, orientation = 'H'):
    ''' algo genetique applique a un liste de slides H ou V exclusivement 
    entrees : taille de la population, population, photos, hyperparametre s de selection, 
    liste des resultats stockes de l'algo genetique, l'orientation des photos de la population
    sortie : nouvelle population
    '''
    scores = []
    #evaluation 
    for i in range(taille):
        scores.append(evaluation(population[i],photos))
    #rangement par score decroissant pour matcher les probabilites
    population = [individu for score,individu in sorted(zip(scores,population),reverse=True)]
    probas = rangement_par_qualite(len(scores),s)
    population2 = []
    for i in range(taille//2):
        #selection
        x,y = selection(population,probas)
        #croisement si photos horizontales
        if orientation == 'H':
            x,y = croisement(x,y)
        #mutation = recherche locale
        x,y = algo_descente(x,photos), algo_descente(y,photos)
        population2.append(x)
        population2.append(y)
        scores.append(evaluation(x,photos))
        scores.append(evaluation(y,photos))
    population += population2
    population = [individu for score,individu in sorted(zip(scores,population),reverse=True)][:taille]
    res.append(max(scores))
    return population

def algo_genetique(instance,nb_generation,taille,s,temps_limite=60):
    ''' algo genetique : 
    entrees : instance, nombre de generations, taille de la population, hyperparametre s de selection et un temps limite
    sorties : la meilleure presentation et son score, l'evolution des scores des parties H,V et de leur fusion
    '''
    #initialisation des parametres
    debut = time.time()
    photos,numerosH,numerosV = instance
    H = len(numerosH)
    V = len(numerosV)
    resH = []
    resV = []
    res = []
    cpt = 0
    
    #generation
    populationH,populationV = generer_population(taille,instance)
    #boucler selon nombre de generations ou jusqu'a un temps limite
    while(cpt<nb_generation and (time.time()-debut)<temps_limite):
        if H:
            populationH = next_gen(taille,photos,populationH,s,resH,orientation = 'H')
        if V:
            populationV = next_gen(taille,photos,populationV,s,resV,orientation = 'V')
            
        #fusionner ou non les parties H et V si elles existent ou non
        if H:
            if V:
                res.append(resH[-1]+resV[-1]+score_transition(tags(populationH[0][-1],photos),tags(populationV[0][-1],photos)))
            else:
                res.append(resH[-1])
        else:
            res.append(resV[-1])
        cpt += 1
        if(cpt%10 == 0):
            print("{} generations en {} secs".format(cpt,time.time()-debut))
    presentation = populationH[0] + populationV[0]
    return presentation,res[-1],resH,resV,res
