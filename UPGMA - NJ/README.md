<pre>
Implementation of :
- UPGMA (unweighted pair group method with arithmetic mean) an agglomerative (bottom-up) hierarchical 
clustering method.
- Neighbor Joining an agglomerative (bottom-up) clustering method for the creation of phylogenetic trees.

Both results are in Newick format.
</pre> 

<pre>
M3 = [ 0, 19, 27,  8, 33, 18, 13]
     [19,  0, 31, 18, 36,  1, 13]
     [27, 31,  0, 26, 41, 32, 29]
     [ 8, 18, 26,  0, 31, 17, 14]
     [33, 36, 41, 31,  0, 35, 28]
     [18,  1, 32, 17, 35,  0, 12]
     [13, 13, 29, 14, 28, 12,  0]

UPGMA for M3 with newick format : 
((((A:4.0,D:4.0):4.25,((B:0.5,F:0.5):5.75,G:6.25):2.0):6.25,C:14.5):2.5,E:17.0);
</pre> 
![Alt text](https://github.com/ljp95/master1/blob/master/UPGMA%20-%20NJ/results/UPGMA.PNG)

<pre>
M4 = [0, 2, 4, 6, 6, 8] 
     [2, 0, 4, 6, 6, 8]
     [4, 4, 0, 6, 6, 8]
     [6, 6, 6, 0, 4, 8] 
     [6, 6, 6, 4, 0, 8] 
     [8, 8, 8, 8, 8, 0]

NJ for M4 with newick format : (((A:1.0,B:1.0):1.0,C:2.0):1.0,((D:2.0,E:2.0):1.0,F:5.0):1.0);
</pre> 
![Alt text](https://github.com/ljp95/master1/blob/master/UPGMA%20-%20NJ/results/NJ.PNG)
