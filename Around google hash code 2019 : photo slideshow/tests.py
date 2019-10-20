from around_photo_slideshow import *

''' 
    a : 3 photos
    b : 80000 H
    c : 1000 H ou V
    d : 90000  H ou V
    e : 80000 V
'''

path_a = 'a_example.txt'
path_b = 'b_lovely_landscapes.txt'  
path_c = 'c_memorable_moments.txt'
path_d = 'd_pet_pictures.txt'
path_e = 'e_shiny_selfies.txt'   

#INSTANCE, choose a path
path = path_e
p = 0.01
instance = read(path,p)
photos,numbersH,numbersV = instance

########## Greedy methods
# choose 1 or 2 for greedy methods 1 and 2
number = 2

begin = time.time()
greedy = algo_greedy(instance,greedy=number)
print("greedy : {} in {}secs".format(evaluation(greedy,photos),time.time()-begin))
checking_duplicate(greedy)

########## greedy + descent
begin = time.time()
greedy_ls = algo_descent(greedy,photos)
print("greedy + local search : {} in {}secs".format(evaluation(greedy_ls,photos),time.time()-begin))
checking_duplicate(greedy_ls)


########## genetic
nb_generation = 10
size = 6
s = 0.3
time_limit = 10000

begin = time.time()
genetic,score,resH,resV,res = algo_genetic(instance,nb_generation,size,s,time_limit)
print("genetic : {} in {}secs".format(evaluation(genetic,photos),time.time()-begin))
#plt.plot(list(range(len(res))),res)
checking_duplicate(genetic)

#genetic + descent
begin = time.time()
genetic_ls = algo_descent(genetic,photos)
print("genetic + Local search : {} in {}secs".format(evaluation(genetic_ls,photos),time.time()-begin))
checking_duplicate(genetic_ls)


########## descent only
descent = presentation_simple(instance)
time_limit = 15
res = []
max_iter = 100000
begin = time.time()
while(time.time()-begin<time_limit):
    descent = algo_descent(descent,photos,max_iter=max_iter)
    res.append(evaluation(descent,photos))
print("Local search : {} in {}secs".format(evaluation(descent,photos),time.time()-begin))
#plt.plot(list(range(len(res))),res)
checking_duplicate(descent)

############################### Linear programming ##############################
# 
##INSTANCE
##path = 'b_lovely_landscapes.txt'  
#p = 0.01
#photos,numbersH,numbersV = read(path,p)     
#instance = [photos,numbersH,numbersV] 
#n = len(photos)
#
##PLNE
#integer = True
#sub_tours = 1
#
#begin = time.time()
#model,X = LP(photos,integer,sub_tours)
#print("PLNE : {} en {}secs".format(model.objVal,time.time()-begin))
#
##PL et rounding
#integer = False
#sub_tours = 0
#
#begin = time.time()
#model,X = PL(photos,integer,sub_tours)
#solution_pl = update(X,n)
#presentation_lp = rounding(solution_pl,photos)
#print("rounding LP : {} in {}secs".format(evaluation(presentation_lp,photos),time.time()-begin))
#checking_duplicate(presentation_lp)
#
