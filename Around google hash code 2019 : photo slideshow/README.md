<pre>
In the context of Problem resolution course

The module gurobipy need to be installed for linear programming !

We denote p as the percentage of data we read.
Data "b" : 80000 horizontal photos
     "d" : 30000 horizontal and 60000 vertical photos
     "e" : 80000 vertical photos

we refer to photoH and photoV as respectively horizontal photo and vertical photo.
</pre>

<pre>
First greedy method : 
We look for the photo which maximize the transition, if it's a photoH, we have our slide.
Else we look for a second photoV which maximize separately the transition. 
In this case we loop on all photos then on all photosV.
Notice that it probably decreases the transition.

Second greedy method : 
We look for photosH only until there are no more then the photosV.
As the algorithm won't look for all photos (except if there is just one type of photos)
and that we never loop a second time on photosV,the second greedy method is much faster (x4) 
and still keep around 90% of first greedy method.
</pre>

<p float="left">
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_g1_g2.png />
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_times.png/> 
</p>

<pre>
Stochastic descent : randomly choose the following swaps :
	- swap randomly two slides 
	- permute randomly photosV between two slides
  
First greedy method on the left, the second one on the right.
Notice that the local search in the second greedy method has more margin to catch up.
At a certain point, it is naturally more and more difficult to get better score with local search.
</pre>
<p float="left">
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_g1_rl_01.png />
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_g2_rl_01.png> 
</p>
<p float="left">
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_g1_rl_05.png />
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_g2_rl_05.png> 
</p>

<pre>
Genetic algorithm :
 - Initialization : generate a population randomly. Separate a individual in two : a photosH and a photosV parts
 - Selection : we use a ranking by quality. 
 Then the probability to be chosen is pb((1-pb)^rank) with pb a hyperparameter fixed to 0.3
 - Crossover : on a point only with photosH. No solution was found for photosV.
 - Mutation : we use the local search with stochastic descent. We apply it separately on photosH and photosV
</pre>
<p float="left">
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_gen_01.png />
  <img src=https://github.com/ljp95/master1/blob/master/Around%20google%20hash%20code%202019%20:%20photo%20slideshow/results/d_gen_05.png> 
</p>

<pre>
Integer linear programming
We implemented the MTZ constraints and "par les flots" constraints.
As it looks for a optimal solution, it is very slow even with very small data.
</pre>


