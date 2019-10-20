
""" Created by ljp95 and CAI Bailin on 05.08.2019 """

import numpy as np
import random
import time
import copy
import gurobipy as grb

############################# Some functions needed #############################
def read(path,p):
    ''' read p (percentage) of a file
    photo format : [orientation, number of tags, set of tags, photo number]
    input : path and p
    output : - list composed of lists of photos
             - list of photosH numbers
             - list of photosV numbers
    '''
    # parameters initialization
    f = open(path,mode = 'r')
    n = int(f.readline());
    photos = []
    numbersH = []
    numbersV = []
    for i in range(int(n*p)):
        # shaping the line
        line = f.readline()
        line = line.replace("\n", "")
        line = line.split(' ')
        photo = [line[0],int(line[1]),set(line[2:]),i]
        photos.append(photo)
        
        # adding to the appropriate list according to the photo orientation
        if photo[0] == 'H':
            numbersH.append(i)
        if photo[0] == 'V':
            numbersV.append(i)  
    f.close()
    return [photos,numbersH,numbersV]
               
def presentation_simple(instance):
    """ Return simple presentation : all photosH then all photosV in apparition order
    input : an instance which is a list of photos, numbersH and numbersV
    output : presentation in the form of a list of lists of photos numbers
    """
    photos,numbersH,numbersV = instance
    presentation = []
    for i in numbersH:
        presentation.append([i])
    for i in range(len(numbersV)//2):
        presentation.append([numbersV[2*i],numbersV[2*i+1]])
    return presentation
    
def write(presentation,filename):
    """ Produce a file according to the requested format :
        number of slides, photos numbers of each slide
        input : presentation, name of the file
        output : /
    """
    f = open(filename,'w')
    f.writelines([str(len(presentation)),'\n'])
    for slide in presentation[:-1]:
        if(len(slide) != 1):
            f.writelines([str(slide[0]), ' ', str(slide[1]), '\n'])
        else:
            f.writelines([str(slide[0]),'\n'])
    slide = presentation[-1]
    
    #to avoid the writing a line break at the end
    if(len(slide) != 1):
        f.writelines([str(slide[0]), ' ', str(slide[1])])
    else:
        f.writelines([str(slide[0])])
    f.close()
    return

def score_transition(tags1,tags2):
    """ Return the transition score between the tags of two slides
    input : two sets of tags
    output : transition score
    """
    inter = len(tags1.intersection(tags2))
    return min(inter,len(tags1)-inter,len(tags2)-inter)
            
def tags(slide,photos):
    """ Return a set of tags according the slide's photos 
    input : a slide, a list of photos numbers
    output : a set of tags
    """
    if len(slide) == 1:
        return photos[slide[0]][2]
    else:
        return photos[slide[0]][2].union(photos[slide[1]][2])

def evaluation(presentation,photos):
    """ Return total score
    input : presentation
    output : score of the presentation
    """
    score = 0
    tags1 = tags(presentation[0],photos)
    for i in range(1,len(presentation)-1):
        tags2 = tags(presentation[i],photos)
        score += score_transition(tags1,tags2)
        tags1 = tags2
    return score

def checking_duplicate(presentation):
    """ Checking if there is duplicate in the presentation
    input : presentation
    output : /
    """
    numbers_used = set()
    for slide in presentation:
        if len(slide) == 1:
            if slide[0] in numbers_used:
                print("{} duplicate".format(slide[0]))
            else:
                numbers_used.add(slide[0])
        else:
            if slide[0] in numbers_used:
                print("{} duplicate".format(slide[0]))
            else:
                numbers_used.add(slide[0])
                if slide[1] in numbers_used:
                    print("{} dulicate".format(slide[0]))
                else:
                    numbers_used.add(slide[1])
    print("No duplicate\n")

################################ Greedy methods ###############################
def greedy_init(photos,numbersH_free,numbersV_free):
    """ Return the first slide randomly chosen of the greedy method
    Can handle no photosV and no photosH
    input : photos, free photosH and photosV numbers
    output : a slide
    """
    if len(numbersV_free)>=2:
        if len(numbersH_free) != 0:
            if(random.random()<0.5):
                slide = [random.choice(numbersH_free)]
            else:
                slide = random.sample(numbersV_free,2)
        else:
            slide = random.sample(numbersV_free,2)
    else:
        slide = [random.choice(numbersH_free)]
    return slide

def greedy_init2(photos,numbersH_free,numbersV_free):
    """ Return the first slide of the second greedy method (faster)
    Always favor photosH if possible
    input : photos, free photosH and photosV numbers
    output : a slide
    """
    if(len(numbersH_free)):
        slide = [random.choice(numbersH_free)]
    else:
        slide = random.sample(numbersV_free,2)
    return slide

def best_photo(photos,numeros_dispo,tags1,photo1=None):
    """ Return the photo which maximize the transition among all photos
    If photo1 == None, we look for a photoH or photoV else we look for a photoV to complete the slide
    input : photos, free photos numbers, the set of tags of the previous slide, photo1 if we need to complete
    output : a photo and the current score transition
    """
    if photo1 == None:
        current_score = -1
        for i in numeros_dispo:
            tags2 = photos[i][2]
            score = score_transition(tags1,tags2)
            if(score>current_score):
                current_score = score
                photo1 = i
        return photo1,current_score
    else:
        # only difference with above : to compute the transition we take tags([photo1,photo])
        current_score = -1
        photo2 = None
        for i in numeros_dispo:
            tags2 = tags([photo1,i],photos)
            score = score_transition(tags1,tags2)
            if(score>current_score):
                current_score = score
                photo2 = i
        return photo2,current_score
    
def best_slide1(photos,numbersH_free,numbersV_free,tags1):
    """ Return the slide which maximize the transition
    Look for the photo which maximize the transition
    If it's a photoH, we have maximization
    else it's a photoV so we look for a second photoV to complete but we don't try to compare with the photoH
    as adding a second photoV can make the transition score decreased.
    input : photos, free photosV and photosH numbers, the set of tags of the previous slide
    output : two photos H or V or None"""
    photoH,scoreH = best_photo(photos,numbersH_free,tags1)
    photoV,scoreV = best_photo(photos,numbersV_free,tags1)
    
    if scoreH>=scoreV:
        return photoH,None
    
    #case of best photo is a photoV
    else:
        #remove the photoV of the possibilities
        numbersV_free = list(set(numbersV_free)-set([photoV]))
        photoV2,tmp = best_photo(photos,numbersV_free,tags1,photoV)
    if photoV2 == None:
        return photoH,None
    return photoV,photoV2

def best_slideV(photos,numbersV_free,tags1):
    """ Return the two photosV which maximize separately the transition
    We are using only one loop instead of revisiting the photosV as in the first greedy method
    input : photos, free numbersV and the set of tags of the previous slide
    """
    score1,score2 = -1,-1
    photo1,photo2 = None,None
    for i in numbersV_free:
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
    
def best_slide2(photos,numbersH_free,numbersV_free,tags1):
    """ Return the slide which maximize the transition favoring photosH
    if there are not photosH anymore, we look for two photosV which maximize separately the transition
    input : photos, free photosH and photosV numbers, set of tags of the previous slide
    output : two photosH or photosV or None
    """
    if(numbersH_free):
        photoH,scoreH = best_photo(photos,numbersH_free,tags1)
        return photoH,None
    else:
        photo1,photo2 = best_slideV(photos,numbersV_free,tags1)
    # if there only one photoV i.e photo2 == None, the slide is not possible so we return photo1 = None
    if photo2 == None:
        return None,None
    return photo1,photo2
    
def maj(presentation,photos,numbersH_free,numbersV_free,slide):
    """ update the presentation, the free photos numbers depending on the new slide
    input : presentation, free photosH and photosV numbers and the new slide
    output : set of tags of the new slide
    """
    presentation.append(slide)
    if len(slide) == 2:
        numbersV_free.remove(slide[0])
        numbersV_free.remove(slide[1])
    else:
        numbersH_free.remove(slide[0])
    return tags(slide,photos)

def algo_glouton(instance,temps_limite = 100000,glouton=2):
    """ Return a greedy presentation according to the maximizer function chosen
    input : an instance, a time limit, the number of the greedy method chosen
    output : a presentation
    """

    #parameters initialization
    debut = time.time()
    photos,numbersH,numbersV = instance
    numbersH_free,numbersV_free = copy.deepcopy(numbersH),copy.deepcopy(numbersV)
    presentation = []
    
    #first slide and maximizer function according to the greedy method chosen
    if(glouton == 2):
        maximizer = best_slide2
        slide1 = greedy_init2(photos,numbersH_free,numbersV_free)
    else: 
        maximizer = best_slide1
        slide1 = greedy_init(photos,numbersH_free,numbersV_free)
    
    #update the presentation, free numbers and the current tag
    tags1 = maj(presentation,photos,numbersH_free,numbersV_free,slide1)
    
    #break the loop if time limit exceeded or no right slide <=> photo1 == None
    while(time.time()-debut < temps_limite):
        #looking for the next maximizer slide
        photo1,photo2 = maximizer(photos,numbersH_free,numbersV_free,tags1)
        #break if no free slide
        if photo1 == None:
            break
        #put the photos in the form of a slide/list
        else:
            if photo2 != None:
                slide2 = [photo1,photo2]
            else:
                slide2 = [photo1]
        #update the presentation, free numbers and the current tag
        tags1 = maj(presentation,photos,numbersH_free,numbersV_free,slide2)
        if(len(presentation)%100 == 0):
            print("{} slides in {} secs".format(len(presentation),time.time()-debut))
    return presentation

############################## Stochastic descent #############################
def score_neighbour(pres,photos,ind1,ind2):
    """ Return the sum of transition scores of the slides affected by the swap between index1 and index2 in the presentation
    input : presenation, the two indexs
    output : new transition score of slides affected by the swap between index1 and index2
    """
    score = 0
    for i in range(2):
        # i = 0 : case of transition score between index-1 and index
        # i = 1 : case of transition score between index and index+1
        score += score_transition(tags(pres[ind1-1+i],photos),tags(pres[ind1+i],photos)) 
        score += score_transition(tags(pres[ind2-1+i],photos),tags(pres[ind2+i],photos))
    return score

def algo_descent(pres,photos,max_iter = None):
    """ Stochastic descent between swaping randomly two slides and permuting photosV of the two slides
    input : presentation and a number of iterations
    output : presentation after stochastic descent
    """

    # parameters initialization
    if max_iter == None:
        max_iter = len(pres)//2
    score = 0
    score_desc = 0
    numbers_slide = list(range(1,len(pres)-1))
    numbers_slideV = [i for i in range(1,len(pres)-1) if len(pres[i]) == 2]
    
    # if no photosV, we just swap the slides else 50/50
    if numbers_slideV != []:
        p = 0.5
    else:
        p = 1    
    for i in range(max_iter):
        r = random.random()
        if(r<p):
            """ Compute neighbors scores with and without the the swap of the two slides
            then the presentation is modified """
            [ind1,ind2] = random.sample(numbers_slide,2)
            score = score_neighbours(pres,photos,ind1,ind2)
            pres[ind1],pres[ind2] = pres[ind2],pres[ind1]
            score_desc = score_neighbours(pres,photos,ind1,ind2)

            """ swap again if the neighbor score of the stochastic descent is worse
            else update the photos numbers if it's between photosH and photosV slide.
            If it's between photosV, it doesn't change the numbers """
            
            if(score_desc<score):
                pres[ind1],pres[ind2] = pres[ind2],pres[ind1]
            
            else:
                if(len(pres[ind1]) == 2 and len(pres[ind2]) == 1):
                    numbers_slideV.append(ind1)
                    numbers_slideV.remove(ind2)
                else:
                    if(len(pres[ind1]) == 1 and len(pres[ind2]) == 2):
                        numbers_slideV.append(ind2)
                        numbers_slideV.remove(ind1)
        else:
            # compute neighbor score with and without the swap of photosV between two slides
            [ind1,ind2] = random.sample(numbers_slideV,2)
            score = score_neighbours(pres,photos,ind1,ind2)
            [ind11,ind22] = random.sample([0,0,1,1],2)
            pres[ind1][ind11],pres[ind2][ind22] = pres[ind2][ind22],pres[ind1][ind11]
            score_desc = score_neighbours(pres,photos,ind1,ind2)
            
            
            # swap again if the neighbor score of the stochastic descent is worse
            if(score_desc<score):
                pres[ind1][ind11],pres[ind2][ind22] = pres[ind2][ind22],pres[ind1][ind11]
    return pres

############################## Genetic algorithm ##############################
def generate_population(size, instance):
    """ Generation of a list of simple presentations i.e of slides H then V by apparition order
    We separate each individual in two, a part H and a part V
    input : size of the population, instance
    output : two lists of population, one for H another for V
    """
    
    # parameters initialization
    photos,numbersH,numbersV = instance    
    populationH = []
    populationV = []
    numbersH_random,numbersV_random = numbersH.copy(), numbersV.copy()
    
    for i in range(size):
        np.random.shuffle(numbersH_random)
        np.random.shuffle(numbersV_random)
        populationH.append(presentation_simple([photos,numbersH_random,[]]))
        populationV.append(presentation_simple([photos,[],numbersV_random]))
    return populationH,populationV

def selection(population,probas):
    """ Return two individuals of the population according to their probability to be chosen
    input : population and a list of probabilities of each individual
    output : two individuals
    """
    ind1,ind2 = np.random.choice(range(len(population)),2,replace=False,p=probas)
    return population[ind1],population[ind2]

def ranking_by_quality(n,p):
    """ Return the probability of each individual according to their rank : p((1-p)^rank)
    input : size of the population, hyperparameter p
    output : list of probability p((1-p)^rank) directly in decreasing order """ 
    probas = p*np.power(1-p,np.arange(n))
    probas /= probas.sum()
    return probas

def crossover(x,y):
    """ Return two sons from x and y
    crossover on point only for photosH, no suggestion for photosV crossover
    input : parents x and y
    output : two sons z1 and z2
    """
    # parameters initialization
    pos = random.randint(1,len(x)-1)
    z1 = x[pos:]
    z2 = y[pos:]
    
    numbers1 = [slide[0] for slide in z1]
    numbers2 = [slide[0] for slide in z2]
    
    for slide in y:
        if slide[0] not in numbers1:
            z1.append(slide)
    for slide in x:
        if slide[0] not in numbers2:
            z2.append(slide)
    return z1,z2

def next_gen(size, photos, population, s, res, orientation = 'H'):
    """ genetic algorithm applied to a list photosV or photosH slides exclusively
    input : size of the population, population, photos, hyperparameter p for selection,
    list of results stored from the genetic algorithm, the orientation of population's photos
    output : new population
    """
    scores = []
    #evaluation 
    for i in range(size):
        scores.append(evaluation(population[i],photos))
    #ranking by decreasing scores to match the probabilities
    population = [individu for score,individu in sorted(zip(scores,population),reverse=True)]
    probas = ranking_by_quality(len(scores),s)
    population2 = []
    for i in range(size//2):
        #selection
        x,y = selection(population,probas)
        #crossover if photosH
        if orientation == 'H':
            x,y = crossover(x,y)
        #mutation = local search
        x,y = algo_descent(x,photos), algo_descent(y,photos)
        population2.append(x)
        population2.append(y)
        scores.append(evaluation(x,photos))
        scores.append(evaluation(y,photos))
    population += population2
    population = [individu for score,individu in sorted(zip(scores,population),reverse=True)][:size]
    res.append(max(scores))
    return population

def algo_genetic(instance,nb_generation,size,s,temps_limite=60):
    """ genetic algorithm
    input : instance, number of generations, size of the population, hyperparameter s for selection and a time limit
    output : the best presentation and his score, the scores evolution of part H, V and their merging
    """
    # parameters initialization
    begin = time.time()
    photos,numbersH,numbersV = instance
    H = len(numbersH)
    V = len(numbersV)
    resH = []
    resV = []
    res = []
    counter = 0
    
    # generation
    populationH,populationV = generate_population(size,instance)
    # looping according to the number of generations or time limit
    while(counter<nb_generation and (time.time()-begin)<temps_limite):
        if H:
            populationH = next_gen(size,photos,populationH,s,resH,orientation = 'H')
        if V:
            populationV = next_gen(size,photos,populationV,s,resV,orientation = 'V')
            
        # merging or not the H and V if they exist
        if H:
            if V:
                res.append(resH[-1]+resV[-1]+score_transition(tags(populationH[0][-1],photos),tags(populationV[0][-1],photos)))
            else:
                res.append(resH[-1])
        else:
            res.append(resV[-1])
        counter += 1
        if(counter%10 == 0):
            print("{} generations in {} secs".format(counter,time.time()-begin))
    presentation = populationH[0] + populationV[0]
    return presentation,res[-1],resH,resV,res


############################ Integer linear programming ########################
    
def PL(photos,integer = False,sub_tours = 0):
    """ linear programming
    input : photos, boolean for using integer or not
    sub_tours : 0 if without, 1 for MTZ, 2 "par les flots"
    output : gurobi model, gurobi solution
    """
    #parameters initialization
    model = grb.Model(name = 'Model')
    n = len(photos)
    V = range(n)

    ### Regarding X
    ## variables xij integers or continuous in a list of lists X
    if integer:
        X = [[model.addVar(vtype=grb.GRB.BINARY,lb=0,ub=1) for i in V]for j in V]
    else:
        X = [[model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,ub=1) for i in V]for j in V]
        # constraints if x continous xij + xji <= 1
        for i in V:
            for j in range(i+1,n):
                model.addConstr(X[i][j]+X[j][i] <= 1)

    ##constraints
    #x[i,i] == 0
    for i in V:
        model.addConstr(X[i][i] == 0)
    # only 1 edge to i
    for i in V:
        model.addConstr((grb.quicksum(X[j][i] for j in V)) == 1)
    # only 1 edge from i
    for i in V:
        model.addConstr((grb.quicksum(X[i][j] for j in V)) == 1)
    
    #no subtours constraints
    if sub_tours == 0:
        pass
    else:
        ## Miller-Tucker-Zemlin subtours elimination
        if sub_tours == 1:   
            #adding n variables u
            if integer:
                U = [model.addVar(vtype=grb.GRB.INTEGER,ub=n) for i in V]
            else:
                U = [model.addVar(vtype=grb.GRB.CONTINUOUS,ub=n) for i in V]
                
            #constraints on possible u values
            for i in range(1,n):
                model.addConstr(U[i] >= 2)
            model.addConstr(U[0] == 1)
            
            #adding others constraints MTZ : ui-uj+1 <= n(1-xij) with i!=j
            for i in range(1,n):
                for j in range(1,n):
                    if i!=j:
                        model.addConstr(U[i]-U[j]+1 <= n*(1-X[i][j]))
                        
        ##"interdiction de sous-tours par les flots"
        else:
            if sub_tours == 2:
                #adding the n^2 variables z
                if integer:
                    Z = [[model.addVar(vtype=grb.GRB.INTEGER,lb=0) for i in V] for j in V]
                else:
                    Z = [[model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0) for i in V] for j in V]
                    
                #constraints on z values
                for i in range(1,n):
                    model.addConstr(Z[i][0] == 0)
                for i in V:
                    model.addConstr(Z[i][i] == 0)
                    
                #sum over j of z0j = n-1
                model.addConstr(grb.quicksum(Z[0][j] for j in V) == n-1)
                
                #sum over j zij+1 = sum over j zji except j=0 and/or j=i cases
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
                            
    #Maximization goal
    objective = grb.LinExpr()
    objective = 0
    for i in V:
        for j in V:
            objective += X[i][j] * score_transition(tags([i],photos),tags([j],photos))
    model.ModelSense = grb.GRB.MAXIMIZE
    model.setObjective(objective)
    #optimization
    model.optimize()
    return model,model.X

############################  LP and rounding  ################################
def shaping(X,n):
    """ Shaping the gurobi solution from vectoriel form to a list of list
    input : solution X from gurobi, number of photos
    output : solution of the linear programming shaped as the list of list xij ranked by decreasing order
    """
    V = list(range(n))
    solution_lp = []
    for i in V:
        for j in V:
            value_x = X[i*(n-1)+j]
            if value_x:
                solution_lp.append([i,j,X[(n-1) * i + j]])
                #solution_lp.append([i,j,score_transition(tags([photos[i]]),tags([photos[j]]))])
    solution_lp = sorted(solution_lp, key = lambda x:x[-1],reverse = True)
    return solution_lp

def rounding(solution_lp,photos):
    """ Rounding the linear programming
    input : solution of the linear programming with adequate shape, photos
    output : a presentation
    
    Method : - first build the subchains by taking edge in decreasing order,
               keep them if there is only one edge which arrives to and only one edge which comes from
               each vertex and if no cycle is made
             - link the subchains to get a hamiltonian path
    """
    #parameters initialization
    n = len(photos)
    starts = set()
    arrivals = set()
    chain = [None for i in range(n)]
    start_of_end = dict()
    end_of_start = dict()

    ##Building the subchains
    for k in range(len(solution_lp)):
        # for each xij chosen by decreasing order
        [i,j,value] = solution_lp[k]
        if i == 18 or j == 18:
            break
        # each vertex is the start and the end of a edge at most
        # we track the start and end of each subchain created  
        if i not in starts and j not in arrivals:
            starts.add(i)
            arrivals.add(j)
            chain[i] = j
            
            #We got 4 cases
            if i in start_of_end:
                if j in end_of_start:
                    # (i,j) is at the start of a chain1 and at the end of chain2
                    # if chain1 != chain2 update all chains affected
                    if start_of_end[i] != j:
                        lien_i = start_of_end.pop(i)
                        lien_j = end_of_start.pop(j)
                        start_of_end[lien_j] = lien_i
                        end_of_start[lien_i] = lien_j
                    #else undo all adding made
                    else:
                        chain[i] = None
                        starts.remove(i)
                        arrivals.remove(j)
                else:
                    #(i,j) is only at the end of existing chain
                    lien = start_of_end.pop(i)
                    end_of_start[lien] = j
                    start_of_end[j] = lien
            else:
                if j in end_of_start:
                    #(i,j) is only at the start of existing chain
                    lien = end_of_start.pop(j)
                    end_of_start[i] = lien
                    start_of_end[lien] = i
                else:
                    #(i,j)is a new chain
                    start_of_end[j] = i
                    end_of_start[i] = j
    
    # parameters initialization
    available = list(set(range(n)).difference(arrivals))
    if available:
        index = random.choice(available)
        presentation_lp = [[index]]
        available.remove(index)
    else:
        index = random.choice(list(range(n)))
        presentation_lp = [[index]]
    ## Building a hamiltonian path
    while len(presentation_pl) != n:
        # if the considered vertex i is not part of a chain
        if chain[index] == None:
            # create an random edge from i to a available vertex j
            index2 = random.choice(available)
            while index2 == index:
                index2 = random.choice(available)
            #update available and the chain
            available.remove(index2)
            chain[index] = index
        # add the arrival vertex j to the presentation and consider the vertex i = vertex j
        index = chain[index]
        presentation_pl.append([index])
    return presentation_pl
            