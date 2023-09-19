# MCS: Efficient Computational Algorithms SA 2023-2024
Michael Multerer, Davide Baroli, Valentina Ghidini and Sara Avesani, Jacopo Quizi.

##
General References:

* IEEE Computing in Science & Engineering [Volume: 2, Issue: 1, Jan.-Feb. 2000](
https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=17639&punumber=5992).
* A. Townsend [The top 10 Algorithms from the 20th Century](https://pi.math.cornell.edu/~ajt/presentations/TopTenAlgorithms.pdf).
* B. Korte and J. Vygen. Combinatorial Optimization.
* T. H. Cormen, C. E. Leierson, R. L. Rivest, and C. Stein. Introduction to Algorithms.
* G. H. Golub, C. F. Van Loan. Matrix Computations.
* H. Harbrecht and M. Multerer. Algorithmische Mathematik (in German).
##
<p align="center">
<img src="https://thekidshouldseethis.com/wp-content/uploads/2012/11/cheetahs-on-the-edge.jpg" width=60% height=60%></img>
</p>



## 1. Spectral Clustering (Supervisor: Sara)
Graph partitioning is heavily used in domain decomposition and social community detection. If the considered graph is embedded into
a coordinate system, there exist simpler methods, for example recursive coordinate bisection and inertial partitioning. However, these
methods totally fail for hypergraphs. This is where the spectral clustering method comes in, which also work for general graphs
without nodal coordinates. The method is inspired by a vibrating string: The label of a node is determined by the sign of the
corresponding element in the second smallest eigenvector, also called Fiedler eigenvector, of the graph Laplacian matrix. 

### References:

* M. Fiedler. Algebraic connectivity of graphs.
* A. Pothen, H. D. Simon, and K. Liou. Partitioning sparse matrices with eigenvectors of graphs.

## 2. Nested Dissection (Supervisor: Michael)
The Cholesky decomposition is a direct solver for symmetric positive definite system $Ax=b$ and corresponds to a 
particular version of the Gaussian elimination. However, even for a sparse matrix, the resulting lower triangular matrix $L$ can 
potentially be dense due to newly added non-zero entries (fill-in), which increase the amount of storage and computation.
The fill-in depends on the order in which variables are eliminated. Nested dissection is a fill-in reducing ordering based on
a graph model. One can think of a sparse matrix as a connectivity matrix and therefore as a graph. The graph is recursively splitted
into pieces using proper separators, numbering the nodes in separators last. For certain classes of graphs, cost bounds
on the corresponding matrix factorization can be proven.

### References:

* A. George. Nested Dissection of a Regular Finite Element Mesh.

## 3. Strassen Algorithm (Supervisor: Sara)
Given two square $n\times n$ matrices, the naive method for computing the matrix multiplication has cost $O(n^3)$. However, it can
be done in a better way using Strassen's algorithm which is a divide and conquer method and reuses intermediate products.
As a consequence, the cost of the matrix factorization is reduced to approximately $O(n^{2.8})$. Based on this observation,
there were ever since efforts to further reduce the cost of the matrix multiplication, a bound proven in 2020 is $O(n^{2.37286})$

### References:

* V. Strassen. Gaussian Elimination is not Optimal.
* J. Alman and V. V. Williams. A Refined Laser Method and Faster Matrix Multiplication.





## 4. Approximate Bayesian Computation (ABC)  (Supervisor: Valentina)
In Bayesian statistics, the posterior distribution of a parameter $\theta$ given a dataset $D$ is usually computed using Bayes' rule
$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)} \propto p(D | \theta) p(\theta)$$
where $p(\theta)$ is the prior distribution of the parameter, $p(D)$ the marginal distribution of the data (which usually acts as a normalising constant, and can be omitted) and $p(D|\theta)$ is the likelihood. In many cases, the likelihood $p(D|\theta)$ is either unknown in closed form, or computationally intractable, but nevertheless necessary for posterior inference. Hence the need to resort to approximate computation. To this aim, there exist different classes of algorithms, among which we can find ABC-based ones.

### References:
* <a href="https://allendowney.github.io/ThinkBayes2/chap20.html"> Allen B. Downey; Think Bayes (2021) - Chapter 20 </a>
* <a href="https://pubmed.ncbi.nlm.nih.gov/23341757/"> Sunnaker et al.; Approximate Bayesian computation (2019)</a>




## 5. Gibbs Sampler (Supervisor: Valentina)
The world is not univariate: usually, to model real data, we need to resort to multivariate distributions, and consequently some inferencial techniques require to sample from them. For example, consider a multivariate, p-dimensional parameter $\theta = (\theta_1, \dots, \theta_p)$ and suppose it is necessary to sample from its $ p- $dimensional distribution $p(\theta)$. Then, the Gibbs sampler proposes to do that by iteratively sampling from the corresponding (univariate) conditional distributions $p(\theta_i | \theta_1, \dots, \theta_{i-1},\theta_{i+1}, \dots, \theta_p)$, for $i=1, \dots, p$.
### References:
* <a href="http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Basic/CasellaGeorge1992.pdf"> George Casella, Edward I. George; Explaining the Gibbs Sampler (1992) </a>
* Christian Robert, George Casella; Monte Carlo Statistical Methods (2013) - Chapter 9

## 6. Multilevel Monte Carlo method (Supervisor: Sara)
The Monte Carlo is one of the most common approaches for approximating the expectation of a random variable $X(\omega)$.
From the weak law of large numbers theorem, the required number of samples depends on the variance for a given accuracy to achieve.
If the variance is small, the Monte Carlo method even with few samples can be of high accuracy.
One of the classic methods for variance reduction involves a hierarchy of control variates and is called
multilevel Monte Carlo method. By computing relatively few samples at the high level of the hierarchy,
but lots of samples at the coarse level, we substaintially save in terms of the total computation.

### References:

* S. Heinrich. Multilevel Monte Carlo Methods
* M. B. Giles. Multilevel Monte Carlo Path Simulation.

## 7. Krylov Subspace Iteration Methods (Supervisor: Michael)
Important methods for the numerical solution of eigenvalue problems and large systems of linear equations, such as the Lanczos or the CG method are based on the projection onto a Krylov subspace. Given a matrix $A\in\mathbb{R}^{n\times n}$ and a vector $v\in \mathbb{R}^n, n\in N$, the Krylov space is defined according to $\mathcal{K}_m(A,v):=\{v,Av,\cdots,A^{m-1}\boldsymbol{v}\}$. Of particular interest are the approximation properties of these subspaces and the relation between them.

### References:
* Y. Saad. Numerical methods for large eigenvalue problem.
* G. H. Golub and C. F. van Loan. Matrix Computation.









## 8. Hierarchical Matrices (Supervisor: Michael)
Hierarchical matrices are special matrices $\mathcal{H}\in \mathbb{R}^{n\times n}$, which have blockwise low-rank $k \ll n$ with respect to a special tree-like partitioning $\tau$, the so-called "Block Cluster Tree". A special challenge is posed by the arithmetic of these matrices, for example the addition and multiplication. In the addition, the rank per matrix block can double and must be recompressed accordingly. For the multiplication of hierarchical matrices with even the same block structure, matrix blocks of different sizes need to be combined.

### References:

* S. Börm, L. Grasedyck, W. Hackbusch. Hierarchical Matrices.
* W. Hackbusch. A sparse matrix arithmetic based on on H-Matrices. Part I: Introduction to H-Matrices.


## 9. Fast Fourier Transform (Supervisor: Sara)
Convolution is commonly used in computer vision as a shift equivariant operator. The convolution ${w}\star{v}$ can be computed either as the multiplication of the circulant matrix derived from the weight vector ${w}$ and the signal vector ${v}$, or in the Fourier basis by first computing the element-wise product of their Fourier transforms, and then coming back to the original coordinates via the inverse Fourier transform. The fast Fourier transform (FFT) was invented to bring down the cost to $O(n\log(n))$. 
This cost reduction enabled modern signal processing and is the foundation of the mp3 audio format.
The algorithm was reinvented by J. W. Cooley and J. Tukey independently 160 years later than the first discovery by Carl Friedrich Gauss.

### References:

* J. W. Cooley and J. Tukey. An algorithm for the machine calculation of complex Fourier series.
* M. Bronstein. Deriving convolution from first principles.


## 10. Fast Wavelet Transform (Supervisor: Sara)
Similar to the fast Fourier transform, the fast wavelet transform (FWT) computes a change of basis into a wavelet basis. Different from 
the fast Fourier transform, the FWT can be performed in linear cost with respect to the length of the signal.
It is used in many fields, primarily for signal processing and, consequently, image analysis as a replacement for Fourier transform and discrete cosine transform, with a notable mention going to the work done in JPEG2K. The FWT can also be applied to decompose and filter images.

### References:

* S. Mallat. A wavelet tour of signal processing.


## 11. Sparse Grids (Supervisor: Jacopo-Sara)
Suppose to achieve a required accuracy, we need to employ at least $N$ grid points in the one-dimensional space. With regular grid-based approaches, a straight forward extension to $d$ dimensions leads to $N^d$ grid points to reach the same accuracy. Therefore, regular grid-based methods exhibit the problem of the exponential dependence of dimensionality, i.e., the curse of dimensionality. For the approximation of certain classes of functions, sparse grids overcome the curse of dimensionality and lead to efficient representations.

### References:

* H. J. Bungartz and M. Griebel. Sparse grids.
* https://sparsegrids.org


## 12.  Tensor-trains (Supervisor: Davide)
A tensor is an array with dimensionality more than 2. Because of the curse of dimensionality, challenges are posed by the storage of high-dimensional tensors and the implementation of their arithmetic operations. The tensor-train decomposition is one possible solution, considered as an extension of low rank-approximation of matrices. In this method, one can unfold a tensor recursively by spliting indices into two parts at each step, and perform any low-rank approximation on the resulting 2D matrices. If the low rank $r$ is small, the storage will decrease from $O(n^d)$ to $O(dnr^2)$. The cost of the arithmetic operations reduces dramatically as well, e.g., addition and element-wise multiplication.

### References:

* I. V. Oseledets. Tensor-Train Decomposition.












## 13. Randomized Sketching (Supervisor:Sara )
If you want to multiply an $n \times d$ matrix $X$, with $n>>d$, on the left by an $m \times n$ matrix $\tilde G$ of i.i.d. Gaussian random variables, it is too slow. The idea is to introduce a new randomized $m \times n$ matrix $T$, for which one can compute $T ⋅X$ in only $O(nnz(X)) + \tilde O(m^{1.5} ⋅d^3)$ time, for which the total variation distance between the distributions $T ⋅X$ and $\tilde G ⋅X$ is as small as desired. Here $nnz(X)$ denotes the number of non-zero entries of $X$. Since the total variation distance is small, we can provably use $T ⋅X$ in place of $\tilde G ⋅X$ in any application.

### References:

* https://proceedings.mlr.press/v48/kapralov16.html
* https://arxiv.org/abs/2210.11295



  
  
## Programming Languages for code

<p align="center">
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/python_logo.png"></img>
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/cpp_logo.png"></img>
   <img  height="300" src="https://www.r-project.org/logo/Rlogo.png"></img>
</p>



