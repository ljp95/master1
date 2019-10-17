
def read(path,p):
    ''' lecture de p% d'un fichier 
    photo de forme [orientation, nombre de tags, set de tags, numero de photo]

    entrees : chemin et p% voulu
    sorties : liste composee d'une liste des photos, 
    une liste des numeros des photosH, 
    une liste des numeros des photosV
    '''
    #initialisation des parametres
    f = open(path,mode = 'r')
    n = int(f.readline());
    photos = []
    numerosH = []
    numerosV = []
    for i in range(int(n*p)):
        #mise en forme de la ligne
        line = f.readline()
        line = line.replace("\n", "")
        line = line.split(' ')
        photo = [line[0],int(line[1]),set(line[2:]),i]
        photos.append(photo)
        
        #ajout a la liste appropriee selon l'orientation de la photo
        if photo[0] == 'H':
            numerosH.append(i)
        if photo[0] == 'V':
            numerosV.append(i)  
    f.close()
    return [photos,numerosH,numerosV]
               
def presentation_simple(instance):
    ''' renvoit la presentation simple : toutes les photosH puis toutes les photosV dans l'ordre d'apparition
    entree : instance qui est une liste de photos, numerosH et numerosV
    sortie : presentation sous forme d'une liste de listes de numeros de photos  
    '''
    photos,numerosH,numerosV = instance
    presentation = []
    for i in numerosH:
        presentation.append([i])
    for i in range(len(numerosV)//2):
        presentation.append([numerosV[2*i],numerosV[2*i+1]])
    return presentation
    
def write(presentation,nom_fichier):
    ''' production d'un fichier sous format demande : 
    nombre de slides puis numeros des photos de chaque slide pour chaque ligne
    entrees : presentation, nom du fichier produit voulu
    sortie : /
    '''
    f = open(nom_fichier,'w')
    f.writelines([str(len(presentation)),'\n'])
    for slide in presentation[:-1]:
        if(len(slide) != 1):
            f.writelines([str(slide[0]), ' ', str(slide[1]), '\n'])
        else:
            f.writelines([str(slide[0]),'\n'])
    slide = presentation[-1]
    
    #pour eviter l'ecriture d'un saut de ligne a la fin
    if(len(slide) != 1):
        f.writelines([str(slide[0]), ' ', str(slide[1])])
    else:
        f.writelines([str(slide[0])])
    f.close()
    return

def score_transition(tags1,tags2):
    ''' retourne le score de transition entre les tags de deux slides 
    entrees : deux sets de tags/mots-cles
    sortie : score de transition
    '''
    inter = len(tags1.intersection(tags2))
    return min(inter,len(tags1)-inter,len(tags2)-inter)
            
def tags(slide,photos):
    ''' renvoit un set de tags selon les photos de la slide 
    entree : une slide, une liste de numero(s) de photo
    sortie : un set de tags
    '''
    if len(slide) == 1:
        return photos[slide[0]][2]
    else:
        return photos[slide[0]][2].union(photos[slide[1]][2])

def evaluation(presentation,photos):
    ''' renvoit le score total 
    entree : presentation
    sortie : score de la presentation
    '''
    score = 0
    tags1 = tags(presentation[0],photos)
    for i in range(1,len(presentation)-1):
        tags2 = tags(presentation[i],photos)
        score += score_transition(tags1,tags2)
        tags1 = tags2
    return score

def verif_doublons(presentation):
    ''' verifie s'il y a des doublons dans la presentation
    entree : presentation
    sortie : /
    '''
    numeros_utilises = set()
    for slide in presentation:
        if len(slide) == 1:
            if slide[0] in numeros_utilises:
                print("{} doublon".format(slide[0]))
            else:
                numeros_utilises.add(slide[0])
        else:
            if slide[0] in numeros_utilises:
                print("{} doublon".format(slide[0]))
            else:
                numeros_utilises.add(slide[0])
                if slide[1] in numeros_utilises:
                    print("{} doublon".format(slide[0]))
                else:
                    numeros_utilises.add(slide[1])
    print("Pas de doublon\n")
    return