
"""
Created by ljp95 on 12.7.2018
"""

import pandas as pd
import string

'''A matrix is additive if it's symmetrical, his diagonal is composed of 0 
and all others entries are positive real'''
def verifyAdditivity(matrix):
    for i in range(len(matrix)):
        if(matrix[i][i] != 0):
            return False
        for j in range(i):
            if(matrix[i][j] < 0 or matrix[i][j] != matrix[j][i]):
                return False
    return True

'''A symmetric matrix is ultrametric if for each i,j,k : max(d(i,j),d(j,k),d(k,i)) is not unique'''
def verifyUltrametric(matrix):
    #3 cases of non ultrametric 
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            for k in range(len(matrix)):
                max_ijk = max(matrix[i][j],matrix[j][k],matrix[k][i])
                if(max_ijk != matrix[i][j]):
                    if(max_ijk != matrix[j][k]):
                        #max != d(i,j) and d(j,k)
                        return False
                    else:
                        if(max_ijk != matrix[k][i]):
                            #max != d(i,j) and d(k,i)
                            return False
                else:
                    #max != d(j,k) and d(k,i)
                    if(max_ijk != matrix[j][k] and max_ijk != matrix[k][i]):
                        return False
    return True
    
'''UPGMA is a simple agglomerative (bottom-up) hierarchical clustering method'''
'''The Newick format is a way of describing phylogenetic relationship'''
def nearestClusters(matrix):
    #Checking 'lower triangular'
    Ci = matrix.index[1]
    Cj = matrix.index[0]
    for i in range(2,len(matrix)):
        for j in range(i):
            if(matrix[Ci][Cj] > matrix[matrix.index[i]][matrix.index[j]]):
                Ci = matrix.index[i]
                Cj = matrix.index[j]
    #Placing Ci and Cj in order
    if(Ci > Cj):
        Ci,Cj = Cj,Ci
    return Ci,Cj

def distanceToCk(matrix,Ci,Cj):
    dist_to_Ck = []
    for Cl in matrix.index:
        if(Cl not in Ci + Cj):
            dist_Cl_to_Ck = (matrix[Cl][Ci]*len(Ci)+ matrix[Cl][Cj]*len(Cj)) / (len(Ci)+len(Cj))
            dist_to_Ck.append(dist_Cl_to_Ck)
    return dist_to_Ck        
    
def updateMatrix(matrix,Ck,Ci,Cj):
    #Calculate distance from Ck to all others clusters
    dist_to_Ck = distanceToCk(matrix,Ci,Cj)
    #Removing
    matrix = matrix.drop([Ci],axis=0)
    matrix = matrix.drop([Cj],axis=0)
    matrix = matrix.drop([Ci],axis=1)
    matrix = matrix.drop([Cj],axis=1)
    #Adding
    matrix.loc[Ck] = dist_to_Ck
    dist_to_Ck.append(0)
    matrix.loc[:,Ck] = dist_to_Ck
    return matrix

def updateNewick(newick,Ck,Ci,Cj,branch_length):
    
    '''newick is a dictionary with clusters as keys 
    and [sentence for newick format associated to the key-cluster, his branch's length]
    
    We got 4 cases : if Ci or/and Cj is/are already in the dictionary
    if that's the case, we remove the key(s) and values and add them to another key Ck = Ci + Cj'''
    
    if(Ci in newick):
        Ci_sentence, Ci_branch_length = newick.pop(Ci)
        if(Cj in newick):
            Cj_sentence, Cj_branch_length = newick.pop(Cj)
            newick[Ck]=["({}:{},{}:{})".format(Ci_sentence,branch_length-Ci_branch_length,Cj_sentence,branch_length-Cj_branch_length),branch_length]
        else:
            newick[Ck]=["({}:{},{}:{})".format(Ci_sentence,branch_length-Ci_branch_length,Cj,branch_length),branch_length]
    else:
        if(Cj in newick):
            Cj_sentence, Cj_branch_length = newick.pop(Cj)
            newick[Ck] = ["(}:{},{}:{})".format(Cj_sentence,branch_length-Cj_branch_length,Ci,branch_length-Cj_branch_length),branch_length]
        else:
            newick[Ck]=["({}:{},{}:{})".format(Ci,branch_length,Cj,branch_length),branch_length]
    return newick

def upgma(matrix):
    # Adding letters to rows and columns
    alphabet = list(string.ascii_uppercase[0:len(matrix)])
    matrix = pd.DataFrame(matrix,index = alphabet,columns = alphabet)
    newick = {}
    while(len(matrix) > 2):
        Ci,Cj = nearestClusters(matrix)
        Ck = Ci + Cj
        branch_length = round(matrix[Ci][Cj],2) / 2
        matrix = updateMatrix(matrix,Ck,Ci,Cj)
        newick = updateNewick(newick,Ck,Ci,Cj,branch_length)
#        print(matrix)
#        print(newick)
    #Same as in the while loop but adding ";" to newick
    Ci,Cj = nearestClusters(matrix)
    Ck = Ci + Cj
    branch_length = float(matrix[Ci][Cj]) / 2
    newick = updateNewick(newick,Ck,Ci,Cj,branch_length)
    return newick[Ck][0] + ";"

'''Neighbors Joining is a bottom-up (agglomerative)
 clustering method for the creation of phylogenetic trees
'''
def calculateU(Ci,matrix):
    return matrix[Ci].sum() / (len(matrix)-2)

def calculateQij(Ci,Cj,matrix):
    return matrix[Ci][Cj] - calculateU(Ci,matrix) - calculateU(Cj,matrix)

def clustersWithMinQij(matrix):
    #Checking 'lower triangular'
    Ci,Cj = matrix.index[1],matrix.index[0]
    min_Q = calculateQij(matrix.index[1],matrix.index[0],matrix)
    for i in range(1,len(matrix)):
        for j in range(i):
            Qij = calculateQij(matrix.index[i],matrix.index[j],matrix)
            if(min_Q > Qij):
                min_Q = Qij
                Ci = matrix.index[i]
                Cj = matrix.index[j]
    #Placing Ci and Cj in order
    if(Ci > Cj):
        Ci,Cj = Cj,Ci
    return Ci,Cj

def distanceNJToCk(matrix,Ci,Cj):
    dist_to_Ck = []
    for k in matrix.index:
        if(k not in Ci + Cj):
            dist_k_to_Ck = (float(matrix[Ci][k]) + matrix[Cj][k] - matrix[Ci][Cj]) / 2
            dist_to_Ck.append(dist_k_to_Ck)
    return dist_to_Ck        
    
def updateNJMatrix(matrix,Ck,Ci,Cj):
    #Calculate distance from Ck to all others clusters
    dist_to_Ck = distanceNJToCk(matrix,Ci,Cj)
    #Removing
    matrix = matrix.drop([Ci],axis=0)
    matrix = matrix.drop([Cj],axis=0)
    matrix = matrix.drop([Ci],axis=1)
    matrix = matrix.drop([Cj],axis=1)
    #Adding
    matrix.loc[Ck] = dist_to_Ck
    dist_to_Ck.append(0)
    matrix.loc[:,Ck] = dist_to_Ck
    return matrix
    
def updateNJNewick(newick,Ci,Cj,Ck,dist_Ci_to_Ck,dist_Cj_to_Ck):
    
    '''newick is a dictionary with clusters as keys 
    and sentence for newick format associated to the key-cluster
    
    We got 4 cases : if Ci or/and Cj is/are already in the dictionary
    if that's the case, we remove the key(s) and values and add them to another key Ck = Ci + Cj'''
    
    if(Ci in newick):
        if(Cj in newick):
            newick[Ck] = "({}:{},{}:{})".format(newick.pop(Ci),dist_Ci_to_Ck,newick.pop(Cj),dist_Cj_to_Ck)
        else:
            newick[Ck] = "({}:{},{}:{})".format(newick.pop(Ci),dist_Ci_to_Ck,Cj,dist_Cj_to_Ck)
    else:
        if(Cj in newick):
            newick[Ck] = "({}:{},{}:{})".format(newick.pop(Cj),dist_Cj_to_Ck,Ci,dist_Ci_to_Ck)
        else:
            newick[Ck] = "({}:{},{}:{})".format(Ci,dist_Ci_to_Ck,Cj,dist_Cj_to_Ck)
    return newick

def NeighbourJoining(M4):
    # Adding letters to rows and columns
    alphabet = list(string.ascii_uppercase[0:len(M4)])
    matrix = pd.DataFrame(M4,index = alphabet,columns = alphabet)
    newick = {}
    while(len(matrix) > 2):
        Ci,Cj = clustersWithMinQij(matrix)    
        Ck = Ci + Cj
        ui = calculateU(Ci,matrix)
        uj = calculateU(Cj,matrix)
        dist_Ci_to_Ck = round(matrix[Ci][Cj] + ui - uj,2) / 2
        dist_Cj_to_Ck = round(matrix[Ci][Cj] + uj - ui,2) / 2
        matrix = updateNJMatrix(matrix,Ck,Ci,Cj)
        newick = updateNJNewick(newick,Ci,Cj,Ck,dist_Ci_to_Ck,dist_Cj_to_Ck)
#        print(newick)
#        print(matrix)
    #Same as in the while loop but adding ";" to newick
    Ci,Cj = matrix.index[0],matrix.index[1]   
    Ck = Ci+Cj
    dist_Ci_to_Ck = float(matrix[Ci][Cj])
    newick = updateNJNewick(newick,Ci,Cj,Ck,dist_Ci_to_Ck,dist_Ci_to_Ck)
    return newick[Ck] + ";"

    




    

    
    
    