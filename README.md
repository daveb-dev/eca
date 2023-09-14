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



## 4. Adaptive Cross Approximation (Supervisor: Sara)
The adaptive cross approximation is a method for the approximation of asymptotically smooth functions.
These are bivariate functions $f(x,y)$ whose derivatives apart from the diagonal $x=y$ exhibit a certain decay behavior.
It can be shown that these functions can be partitioned in such a way that individual low-rank regions can be approximated.
The actual algorithm is then based on a partially pivoted Gaussian algorithm with a corresponding adaptive decay criterion.

### References:

* M. Bebendorf. Approximation of boundary element matrices.
* M. Bebendorf. Hierarchical Matrices.

## 5. Approximate Bayesian Computation (ABC)  (Supervisor: Valentina)
In Bayesian statistics, the posterior distribution of a parameter $\theta$ given a dataset $D$ is usually computed using Bayes' rule
$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)} \propto p(D | \theta) p(\theta)$$
where $p(\theta)$ is the prior distribution of the parameter, $p(D)$ the marginal distribution of the data (which usually acts as a normalising constant, and can be omitted) and $p(D|\theta)$ is the likelihood. In many cases, the likelihood $p(D|\theta)$ is either unknown in closed form, or computationally intractable, but nevertheless necessary for posterior inference. Hence the need to resort to approximate computation. To this aim, there exist different classes of algorithms, among which we can find ABC-based ones.

### References:
* <a href="https://allendowney.github.io/ThinkBayes2/chap20.html"> Allen B. Downey; Think Bayes (2021) - Chapter 20 </a>
* <a href="https://pubmed.ncbi.nlm.nih.gov/23341757/"> Sunnaker et al.; Approximate Bayesian computation (2019)</a>

## 6. Inverse Transform Sampling (Supervisor: Valentina)
In a plethora of computational problems, there is a need to simulate from a given distribution. For example, it may be necessary to simulate white noise by sampling from a standard gaussian distribution. But how does one do that? 
An elegant way is to use Inverse Transform Sampling to sample $X \sim F_X(x)$, where $F_X(x) = P(X \leq x)$ is the cumulative distribution function (CDF) of $X$. To use this algorithm, we only need to have $F_X(\cdot)$ invertible and and efficient way to sample from a uniform distribution on $[0,1]$ (spoiler: it exists!).

### References:
* <a href="https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html"> Easy blogpost </a>
* Soon-to-be-found appropriate reference

##7. Acceptance-Rejection Algorithm (Supervisor: Valentina)
As said above (see 7.), in a plethora of computational problems, there is a need to simulate from a given distribution. However, sometimes it is not possible to simulate directly from the target distribution $X \sim F_X(x)$, since it may have some undesirable properties (for example, it may have an uncomputable normalising constant, or its cumulative distribution $F_X(\cdot)$ may not be invertible). In this situation, we can propose observations from an "easier" distribution $G_X(x)$, and then accept them if they are suitable to represent the target distribution we are interested in. This is what Rejection sampling does.

### References:
* <a href="http://www.markirwin.net/stat221/Refs/flury90.pdf"> Bernard D. Flury; Acceptance-rejection sampling made easy (1990) </a>
* Christian Robert, George Casella; Monte Carlo Statistical Methods (2013) - Chapter 2.3

## 8. Gibbs Dampler (Supervisor: Valentina)
The world is not univariate: usually, to model real data, we need to resort to multivariate distributions, and consequently some inferencial techniques require to sample from them. For example, consider a multivariate, p-dimensional parameter $\theta = (\theta_1, \dots, \theta_p)$ and suppose it is necessary to sample from its $ p- $dimensional distribution $p(\theta)$. Then, the Gibbs sampler proposes to do that by iteratively sampling from the corresponding (univariate) conditional distributions $p(\theta_i | \theta_1, \dots, \theta_{i-1},\theta_{i+1}, \dots, \theta_p)$, for $i=1, \dots, p$.
### References:
* <a href="http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Basic/CasellaGeorge1992.pdf"> George Casella, Edward I. George; Explaining the Gibbs Sampler (1992) </a>
* Christian Robert, George Casella; Monte Carlo Statistical Methods (2013) - Chapter 9


## 9. Krylov Subspace Iteration Methods (Supervisor: Michael)
Important methods for the numerical solution of eigenvalue problems and large systems of linear equations, such as the Lanczos or the CG method are based on the projection onto a Krylov subspace. Given a matrix $A\in\mathbb{R}^{n\times n}$ and a vector $v\in \mathbb{R}^n, n\in N$, the Krylov space is defined according to $\mathcal{K}_m(A,v):=\{v,Av,\cdots,A^{m-1}\boldsymbol{v}\}$. Of particular interest are the approximation properties of these subspaces and the relation between them.

### References:
* Y. Saad. Numerical methods for large eigenvalue problem.
* G. H. Golub and C. F. van Loan. Matrix Computation.

## 10. QR Algorithm for Computing Eigenvalues (Supervisor: Davide)
The QR algorithm is one of the most popular iterative methods for computing eigenvalues of general matrices $A$. We initialize $B$ with $A$ and then repeatedly update the matrix $B$ with the product $RQ$, where $Q$ and $R$ are obtained by the QR decomposition of $B$ at the previous step. As the iteration goes, it will eventually converge to an upper triangular matrix of which the diagonal is the eigenvalue vector of the original matrix $A$. Due to the large time complexity of the classical QR algorithm, the shift QR algorithm with Hessenberg reduction is more often used in practice.

### References:
* J. G. F. Francis. The QR Transformation—Part 1 and Part 2.
* G. H. Golub and C. F. van Loan. Matrix Computation.

## 11. Singular Value Decomposition (Supervisor: Davide)
The Singular value decomposition (SVD) of an $n\times m$ matrix $A$ of rank $r$ is the decomposition into a product of three matrices
$$A = U\Sigma V^T,$$
where $U$ is a unitary $n\times n$ matrix, $V$ is the adjoint of a unitary $m\times m$ matrix, and $\Sigma$ is a $n\times m$ matrix in the form 
$$\Sigma = [D,\boldsymbol{0};\boldsymbol{0},\boldsymbol{0}].$$
The $r\times r$ matrix $D$ is a diagonal matrix and its diagonal entries are $\sigma_1,\cdots,\sigma_r$ called singularnvalues. In particular, $\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_r.$ With the help of the SVD, it is possible to calculate the best rank-k approximation of $A$
in the spectral norm as well as in the Frobenius norm.

### References:
* C. Eckart and G. Young. The approximation of one matrix by another of lower rank.


## 12. Randomized Low-rank Approximation (Supervisor: Sara)
A very simple class of low-rank approximations of a matrix is obtained by using the product of the matrix and random vectors. A low-rank approximation can be obtained from the vectors in the image of the matrix. Since the main effort of these methods is dominated by matrix-vector multiplications, these algorithms are usually very fast for sparse matrices. In return, however, only pessimistic error estimates are avaliable, and the actual error is often much better.

### References:
* N. Halko, P. G. Martinsson, J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.

## 13. Pivoted Cholesky Decomposition (Supervisor: Michael)
The pivoted Cholesky decomposition is an extremely efficient algorithm to determine a low-rank approximation of a symmetric, positive semidefinite matrix.  To obtain an approximation of a matrix,  only the diagonal and $m$ columns of the underlying matrix need to be computed. Therefore, the method seems appealing especially for fully populated matrices. In particular, a rigorous posteriori error control is available in the trace norm, so that the method always yields a reliable approximation.

### References:
* H. Harbrecht, M. Multerer, and R. Schneider. On the low-rank approximation by the pivoted Cholesky decomposition.


## 14. Fast Multipole Method (Supervisor: Michael)
The fast multipole method (FMM) is an efficient way to compute the matrix-vector multiplication in $O(n)$ or $O(n\log(n))$ with a bounded error for a particular structured dense $n\times n$ matrix $\Phi$. Such matrices arising out of the $n$-body problem are usually blockwise separable of order k，i.e., $\Phi|_{\tau,\sigma}\approx AB$, where $A$ and $B$ are $n\times k$ matrix and $k\times n$ matrix respectively. Herein,
$\tau$ and $\sigma$ are suitable index sets. The matrix-vector multiplication for this block is thus approximately equivalent to perform $A(Bx)$. Because $k$ is usually $O(1)$, the computaional cost is reduced dramatically. Besides, one of the distinct advantages of the FMM is its rigorous error estimates. 

### References:

* L. Greengard and V. Rokhlin. A Fast Algorithm for Particle Simulations.


## 15. Hierarchical Matrices (Supervisor: Michael)
Hierarchical matrices are special matrices $\mathcal{H}\in \mathbb{R}^{n\times n}$, which have blockwise low-rank $k \ll n$ with respect to a special tree-like partitioning $\tau$, the so-called "Block Cluster Tree". A special challenge is posed by the arithmetic of these matrices, for example the addition and multiplication. In the addition, the rank per matrix block can double and must be recompressed accordingly. For the multiplication of hierarchical matrices with even the same block structure, matrix blocks of different sizes need to be combined.

### References:

* S. Börm, L. Grasedyck, W. Hackbusch. Hierarchical Matrices.
* W. Hackbusch. A sparse matrix arithmetic based on on H-Matrices. Part I: Introduction to H-Matrices.


## 16. Fast Fourier Transform (Supervisor: Sara)
Convolution is commonly used in computer vision as a shift equivariant operator. The convolution ${w}\star{v}$ can be computed either as the multiplication of the circulant matrix derived from the weight vector ${w}$ and the signal vector ${v}$, or in the Fourier basis by first computing the element-wise product of their Fourier transforms, and then coming back to the original coordinates via the inverse Fourier transform. The fast Fourier transform (FFT) was invented to bring down the cost to $O(n\log(n))$. 
This cost reduction enabled modern signal processing and is the foundation of the mp3 audio format.
The algorithm was reinvented by J. W. Cooley and J. Tukey independently 160 years later than the first discovery by Carl Friedrich Gauss.

### References:

* J. W. Cooley and J. Tukey. An algorithm for the machine calculation of complex Fourier series.
* M. Bronstein. Deriving convolution from first principles.


## 17. Fast Wavelet Transform (Supervisor: Sara)
Similar to the fast Fourier transform, the fast wavelet transform (FWT) computes a change of basis into a wavelet basis. Different from 
the fast Fourier transform, the FWT can be performed in linear cost with respect to the length of the signal.
It is used in many fields, primarily for signal processing and, consequently, image analysis as a replacement for Fourier transform and discrete cosine transform, with a notable mention going to the work done in JPEG2K. The FWT can also be applied to decompose and filter images.

### References:

* S. Mallat. A wavelet tour of signal processing.


## 18. Sparse Grids (Supervisor: Jacopo-Sara)
Suppose to achieve a required accuracy, we need to employ at least $N$ grid points in the one-dimensional space. With regular grid-based approaches, a straight forward extension to $d$ dimensions leads to $N^d$ grid points to reach the same accuracy. Therefore, regular grid-based methods exhibit the problem of the exponential dependence of dimensionality, i.e., the curse of dimensionality. For the approximation of certain classes of functions, sparse grids overcome the curse of dimensionality and lead to efficient representations.

### References:

* H. J. Bungartz and M. Griebel. Sparse grids.
* https://sparsegrids.org


## 19.  Tensor-trains (Supervisor: Davide)
A tensor is an array with dimensionality more than 2. Because of the curse of dimensionality, challenges are posed by the storage of high-dimensional tensors and the implementation of their arithmetic operations. The tensor-train decomposition is one possible solution, considered as an extension of low rank-approximation of matrices. In this method, one can unfold a tensor recursively by spliting indices into two parts at each step, and perform any low-rank approximation on the resulting 2D matrices. If the low rank $r$ is small, the storage will decrease from $O(n^d)$ to $O(dnr^2)$. The cost of the arithmetic operations reduces dramatically as well, e.g., addition and element-wise multiplication.

### References:

* I. V. Oseledets. Tensor-Train Decomposition.


## 20.  Proximal Operator (Supervisor: Sara)

Consider an optimization problem of the form 

$$ \min_{x \in X} F(Lx) + G(x) $$
where $F:Y\to\mathbb{R}$ and $G:X\to\mathbb{R}$ are convex functions over Hilbert spaces $X$ and $Y$, whose proximity operators can be computed, and $L:X\to Y$ is a linear operator. 
One should be able to compute efficiently the proximal mapping of $F$ and $G$, defined as: 
$$ \mathrm{Prox}_{ \gamma F}(x) = \mathrm{argmin} _{y} \frac{1}{2} ||x-y||^2  + \gamma F(y) $$ 
(the same definition applies also for $G$). 
Several problems of image analysis could be cast into this framework. For instance, 
to denoise an image $\mathbf{I}$ by minimizing the $L^1$ norm of the gradient of reconstructed image, and $L^2$ norm from the original image. The problem therefore reads
$$\min_{u}\int_\Omega|\nabla u|dx+\frac\lambda2||u-\mathbf{I}||_2^2$$
where $L(u)=\nabla u$ and $F$ is the $L^1$ norm, and $G$ proportional to the $L^2$ distance between $u$ and $\mathbf{I}$.

### References:
* A. Chambolle and T. Pock. A First-order primal-dual algorithm for convex problems with application to imaging. _Journal of Mathematical Imaging and Vision_. Vol. 40, no. 1, 2011


## 21.  Algebraic Multigrid (Supervisor: Sara)
Algebraic Multigrid (AMG) is a numerical technique for the solution of large, sparse linear systems. It is an iterative solver that operates on the matrix representation of the 
linear system, typically without requiring knowledge of some underlying geometry.
AMG leverages a hierarchical approach, where it constructs a hierarchy of finer grid levels, each of which represents a simplified version of the original problem.
At the coarsest level, AMG solves the linear system directly, which is computationally cheaper due to reduced problem size. AMG is known for its excellent scalability and robustness, 
making it well-suited for parallel computing environments and a wide range of problem types.

### References:
* [K. Stüben. Algebraic Multigrid, An Introduction with Applications](https://www.scai.fraunhofer.de/content/dam/scai/de/documents/AllgemeineDokumentensammlung/SchnelleLoeser/SAMG/AMG_Introduction.pdf)
* [J. W. Ruge and K. Stüben. Algebraic Multigrid,  1987, 73-130](
 https://epubs.siam.org/doi/10.1137/1.9781611971057.ch4)


## 22. Multilevel Monte Carlo method (Supervisor: Sara)
The Monte Carlo is one of the most common approaches for approximating the expectation of a random variable $X(\omega)$.
From the weak law of large numbers theorem, the required number of samples depends on the variance for a given accuracy to achieve.
If the variance is small, the Monte Carlo method even with few samples can be of high accuracy.
One of the classic methods for variance reduction involves a hierarchy of control variates and is called
multilevel Monte Carlo method. By computing relatively few samples at the high level of the hierarchy,
but lots of samples at the coarse level, we substaintially save in terms of the total computation.

### References:

* S. Heinrich. Multilevel Monte Carlo Methods
* M. B. Giles. Multilevel Monte Carlo Path Simulation.

  
## Programming Languages for code

<p align="center">
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/python_logo.png"></img>
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/cpp_logo.png"></img>
   <img  height="300" src="https://www.r-project.org/logo/Rlogo.png"></img>
</p>



