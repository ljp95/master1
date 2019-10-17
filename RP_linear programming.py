
import gurobipy as grb
import random
from ex1 import *

######################################## EX5 : PLNE
def PL(photos,entier = False,sous_tours = 0):
    ''' programmation lineaire du probleme
    entrees : photos, binaire entier ou non
    sous_tours : 0 pour sans, 1 pour MTZ, 2 pour les flots
    sorties : le modele gurobi, la solution de gurobi
    '''
    #initialisation des parametres
    model = grb.Model(name = 'Model')
    n = len(photos)
    V = range(n)

    ###Concernant X
    ##variables xij entiers ou continus contenues dans une liste de liste X 
    if entier:
        X = [[model.addVar(vtype=grb.GRB.BINARY,lb=0,ub=1) for i in V]for j in V]
    else:
        X = [[model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,ub=1) for i in V]for j in V]
        #contraintes si x continues xij + xji <= 1
        for i in V:
            for j in range(i+1,n):
                model.addConstr(X[i][j]+X[j][i] <= 1)

    ##contraintes 
    #x[i,i] == 0
    for i in V:
        model.addConstr(X[i][i] == 0)
    #1 seule arete arrivant en i
    for i in V:
        model.addConstr((grb.quicksum(X[j][i] for j in V)) == 1)
    #1 seule arete partant de i
    for i in V:
        model.addConstr((grb.quicksum(X[i][j] for j in V)) == 1)
    
    #pas de contraintes de sous tours
    if sous_tours == 0:
        pass
    else:
        ##interdiction de sous-tours par Miller-Tucker-Zemlin
        if sous_tours == 1:   
            #ajout des n variables u
            if entier:
                U = [model.addVar(vtype=grb.GRB.INTEGER,ub=n) for i in V]
            else:
                U = [model.addVar(vtype=grb.GRB.CONTINUOUS,ub=n) for i in V]
                
            #contraintes sur les valeurs de u possibles
            for i in range(1,n):
                model.addConstr(U[i] >= 2)
            model.addConstr(U[0] == 1)
            
            #ajout des autres contraintes MTZ : ui-uj+1 <= n(1-xij) avec i!=j
            for i in range(1,n):
                for j in range(1,n):
                    if i!=j:
                        model.addConstr(U[i]-U[j]+1 <= n*(1-X[i][j]))
                        
        ##interdiction de sous-tours par les flots
        else:
            if sous_tours == 2:
                #ajout des n^2 variables z
                if entier:
                    Z = [[model.addVar(vtype=grb.GRB.INTEGER,lb=0) for i in V] for j in V]
                else:
                    Z = [[model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0) for i in V] for j in V]
                    
                #contraintes sur les valeurs de z
                for i in range(1,n):
                    model.addConstr(Z[i][0] == 0)
                for i in V:
                    model.addConstr(Z[i][i] == 0)
                    
                #somme sur j de z0j = n-1
                model.addConstr(grb.quicksum(Z[0][j] for j in V) == n-1)
                
                #somme sur j zij+1 = somme sur j zji excepte les cas j=0 et/ou j=i
                for i in range(1,n):
                    sij = 0
                    sji = 0
                    for j in range(1,n):
                        if j!=i:
                            sij += Z[i][j]
                            sji += Z[j][i]
                    model.addConstr(sij +1 == sji + Z[0][i])
                    
                #zij + zji <= (n-1)(xji+xij)
                for i in V:
                    for j in V:
                        if i!=j:
                            model.addConstr(Z[i][j]+ Z[j][i] <= (n-1)*(X[i][j]+X[j][i]))
                            
    #objectif de maximisation
    objectif = grb.LinExpr()
    objectif = 0
    for i in V:
        for j in V:
            objectif += X[i][j] * score_transition(tags([i],photos),tags([j],photos))
    model.ModelSense = grb.GRB.MAXIMIZE
    model.setObjective(objectif)
    #optimisation
    model.optimize()
    return model,model.X

######################################## EX6 PL ET ARRONDI
def mise_en_forme(X,n):
    ''' met en forme la solution gurobi de forme vectorielle a une forme de liste de liste
    entrees : solution X de gurobi, nombre de photos
    sortie : solution du PL sous forme d'une liste de liste xij range par xij decroissant
    '''
    V = list(range(n))
    solution_pl = []
    for i in V:
        for j in V:
            valeur_x = X[i*(n-1)+j]
            if valeur_x:
                solution_pl.append([i,j,X[(n-1) * i + j]])
                #rangement par score de transition ?
                #solution_pl.append([i,j,score_transition(tags([photos[i]]),tags([photos[j]]))])
    solution_pl = sorted(solution_pl, key = lambda x:x[-1],reverse = True)
    return solution_pl

def arrondi(solution_pl,photos):
    ''' methode d'arrondi du programme lineaire
    entrees : solution du pl sous forme adequate, photos
    sortie : une presentation 
    
    Methode : - construire d'abord les sous chaines en prenant les aretes par ordre decroissant,
    les garder si une seule entree et une seule sortie par sommet et si pas de cycle forme
              - relier les sous chaines pour former un chemin hamiltonien
    '''
    #initialisation des parametres
    n = len(photos)
    departs = set()
    arrivees = set()
    chaine = [None for i in range(n)]
    debut_de_fin = dict()
    fin_de_debut = dict()

    ##construction des sous-chaines
    for k in range(len(solution_pl)):
        #pour chaque xij choisi de maniere decroissante
        [i,j,valeur] = solution_pl[k]
        if i == 18 or j == 18:
            break
        #chaque sommet observe un flot entrant et un flot sortant au maximum
        #on traque les debuts et fins de chaque sous_chaines actuellement crees    
        if i not in departs and j not in arrivees:
            departs.add(i)
            arrivees.add(j)
            chaine[i] = j
            
            #quatre cas s'offrent a nous
            if i in debut_de_fin:
                if j in fin_de_debut:
                    #(i,j) est au debut d'une chaine1 et a la fin d'une chaine2
                    #si chaine1 != chaine2 mettre a jour toutes les chaines affectees
                    if debut_de_fin[i] != j:
                        lien_i = debut_de_fin.pop(i)
                        lien_j = fin_de_debut.pop(j)
                        debut_de_fin[lien_j] = lien_i
                        fin_de_debut[lien_i] = lien_j
                    #sinon defaire les ajouts effectues
                    else:
                        chaine[i] = None
                        departs.remove(i)
                        arrivees.remove(j)
                else:
                    #(i,j) est seulement a la fin d'une chaine existante
                    lien = debut_de_fin.pop(i)
                    fin_de_debut[lien] = j
                    debut_de_fin[j] = lien
            else:
                if j in fin_de_debut:
                    #(i,j) est seulement au debut d'une chaine existante
                    lien = fin_de_debut.pop(j)
                    fin_de_debut[i] = lien
                    debut_de_fin[lien] = i
                else:
                    #(i,j) est une nouvelle chaine
                    debut_de_fin[j] = i
                    fin_de_debut[i] = j
    
    #initialisation des parametres
    dispo = list(set(range(n)).difference(arrivees))
    if dispo:
        indice = random.choice(dispo)
        presentation_pl = [[indice]]
        dispo.remove(indice)
    else:
        indice = random.choice(list(range(n)))
        presentation_pl = [[indice]]
    ##construction du chemin hamiltonien
    #boucler tant que presentation non remplie
    while len(presentation_pl) != n:
        #si le sommet_i considere ne fait pas parti d'une chaine
        if chaine[indice] == None:
            #creer une arete au hasard vers un sommet_j disponible
            indice2 = random.choice(dispo)
            while indice2 == indice:
                indice2 = random.choice(dispo)
            #mise a jour adequate
            dispo.remove(indice2)
            chaine[indice] = indice2
        #ajouter le sommet_j d'arrivee a la presentation et considerer le sommet_i = sommet_j
        indice = chaine[indice]
        presentation_pl.append([indice])
    return presentation_pl
            