<pre>
In the context of Machine learning course

I) Lasso impact on USPS
We show how l2 (ridge regression) and l1 (lasso) penalties affect the classifier.
We fix the hyperparameters to get similar scores between the 3 classifiers. 
The ridge regression shrinks the coefficients. 
The lasso on top of that put weights at 0. We get a sparse vector of weights.
</pre>
![Alt text](https://github.com/ljp95/master1/blob/master/Inpainting%20with%20lasso/results/usps.PNG)

<pre>
II) Denoising
We show how a sparse representation affect the results.
</pre>
![](https://github.com/ljp95/master1/blob/master/Inpainting%20with%20lasso/results/denoising.PNG)

<pre>
III) Filling missing part of an image
We compare a naive filling and the filling described in [3] which put a priority value to each point of 
the missing part contour.
</pre>
![](https://github.com/ljp95/master1/blob/master/Inpainting%20with%20lasso/results/filling.PNG)

<pre>
References : 
[1] Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin, Image Inpainting via Sparse Representation
Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP ’09)
[2] Julien Mairal Sparse coding and Dictionnary Learning for Image Analysis INRIA Visual Recognition
and Machine Learning Summer School, 2010
[3] A. Criminisi, P. Perez, K. Toyama Region Filling and Object Removal by Exemplar-Based Image
Inpainting IEEE Transaction on Image Processing (Vol 13-9), 2004
</pre>

