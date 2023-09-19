## 4. Adaptive Cross Approximation (Supervisor: Sara)
The adaptive cross approximation is a method for the approximation of asymptotically smooth functions.
These are bivariate functions $f(x,y)$ whose derivatives apart from the diagonal $x=y$ exhibit a certain decay behavior.
It can be shown that these functions can be partitioned in such a way that individual low-rank regions can be approximated.
The actual algorithm is then based on a partially pivoted Gaussian algorithm with a corresponding adaptive decay criterion.

### References:

* M. Bebendorf. Approximation of boundary element matrices.
* M. Bebendorf. Hierarchical Matrices.

## 6. Inverse Transform Sampling (Supervisor: Valentina)
In a plethora of computational problems, there is a need to simulate from a given distribution. For example, it may be necessary to simulate white noise by sampling from a standard gaussian distribution. But how does one do that? 
An elegant way is to use Inverse Transform Sampling to sample $X \sim F_X(x)$, where $F_X(x) = P(X \leq x)$ is the cumulative distribution function (CDF) of $X$. To use this algorithm, we only need to have $F_X(\cdot)$ invertible and and efficient way to sample from a uniform distribution on $[0,1]$ (spoiler: it exists!).

### References:
* <a href="https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html"> Easy blogpost </a>
* Soon-to-be-found appropriate reference

## 7. Acceptance-Rejection Algorithm (Supervisor: Valentina)
As said above (see 7.), in a plethora of computational problems, there is a need to simulate from a given distribution. However, sometimes it is not possible to simulate directly from the target distribution $X \sim F_X(x)$, since it may have some undesirable properties (for example, it may have an uncomputable normalising constant, or its cumulative distribution $F_X(\cdot)$ may not be invertible). In this situation, we can propose observations from an "easier" distribution $G_X(x)$, and then accept them if they are suitable to represent the target distribution we are interested in. This is what Rejection sampling does.

### References:
* <a href="http://www.markirwin.net/stat221/Refs/flury90.pdf"> Bernard D. Flury; Acceptance-rejection sampling made easy (1990) </a>
* Christian Robert, George Casella; Monte Carlo Statistical Methods (2013) - Chapter 2.3

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

## 13. Pivoted Cholesky Decomposition (Supervisor: Michael)
The pivoted Cholesky decomposition is an extremely efficient algorithm to determine a low-rank approximation of a symmetric, positive semidefinite matrix.  To obtain an approximation of a matrix,  only the diagonal and $m$ columns of the underlying matrix need to be computed. Therefore, the method seems appealing especially for fully populated matrices. In particular, a rigorous posteriori error control is available in the trace norm, so that the method always yields a reliable approximation.

### References:
* H. Harbrecht, M. Multerer, and R. Schneider. On the low-rank approximation by the pivoted Cholesky decomposition.


## 14. Fast Multipole Method (Supervisor: Michael)
The fast multipole method (FMM) is an efficient way to compute the matrix-vector multiplication in $O(n)$ or $O(n\log(n))$ with a bounded error for a particular structured dense $n\times n$ matrix $\Phi$. Such matrices arising out of the $n$-body problem are usually blockwise separable of order k，i.e., $\Phi|_{\tau,\sigma}\approx AB$, where $A$ and $B$ are $n\times k$ matrix and $k\times n$ matrix respectively. Herein,
$\tau$ and $\sigma$ are suitable index sets. The matrix-vector multiplication for this block is thus approximately equivalent to perform $A(Bx)$. Because $k$ is usually $O(1)$, the computaional cost is reduced dramatically. Besides, one of the distinct advantages of the FMM is its rigorous error estimates. 

### References:

* L. Greengard and V. Rokhlin. A Fast Algorithm for Particle Simulations.

## 20.  Proximal Operator (Supervisor: Sara)

Consider an optimization problem of the form 
$$\min_{x \in X} F(Lx) + G(x)$$
where $F:Y\to\mathbb{R}$ and $G:X\to\mathbb{R}$ are convex functions over Hilbert spaces $X$ and $Y$, 
whose proximity operators can be computed, and $L:X\to Y$ is a linear operator. 
One should be able to compute efficiently the proximal mapping of $F$ and $G$, defined as: 

$$ \mathop{\text{Prox}}_{\gamma F}(x) =  \mathop{\text{argmin}}_y \frac{1}{2} {\lVert x-y \rVert}^2  + \gamma F(y) $$

(the same definition applies also for $G$). 
Several problems of image analysis can be cast into this framework. An example is the 
denoising of an image $\mathbf{I}$ by minimizing the $L^1$ norm of the gradient of reconstructed image, and the $L^2$ norm of the original image. The problem then reads
$$\min_{u}\int_\Omega {\lVert \nabla u \rVert} _2 dx+\frac\lambda2 {\lVert u-\mathbf{I} \rVert} _2^2$$
where $L(u)=\nabla u$ and $F$ is the $L^1$ norm, and $G$ proportional to the $L^2$ distance between $u$ and $\mathbf{I}$.

### References:
* A. Chambolle and T. Pock. A First-order primal-dual algorithm for convex problems with application to imaging. _Journal of Mathematical Imaging and Vision_. Vol. 40, no. 1, 2011

## 23. Markov Chain Monte Carlo (Supervisor: Sara)
Markov Chain Monte Carlo (MCMC) stands as a powerful and versatile method employed for the approximate generation of samples from arbitrary probability distributions. This is particularly invaluable when dealing with complex scenarios where the random vector X comprises dependent components. What sets MCMC apart is its ability to navigate these intricate landscapes without necessitating an exact expression of the target probability density function (pdf), often needing only the pdf up to a normalization constant. 
In essence, MCMC constructs a Markov chain where each state represents a potential sample from the desired distribution. By iteratively proposing and accepting/rejecting new states based on their likelihood relative to the current state, the chain eventually converges to a stationary distribution that closely approximates the target distribution. This convergence ensures that the generated samples provide insights and estimates for various statistical and probabilistic analyses.
MCMC's significance extends across diverse fields, enabling Bayesian inference in statistics, facilitating parameter estimation in machine learning, aiding in the simulation of physical systems, and contributing to risk assessment in finance, among many other applications.

### References:

* 
* 

## 25. Wavelet Packets (Supervisor: )
Wavelet packets are a decomposition technique used in signal processing and data analysis, which extends the concept of wavelet transforms. They provide a more flexible and detailed representation of a signal by allowing for a broader range of frequency components to be analyzed. They find applications in various domains, including signal denoising, feature extraction from time-series data, image compression, audio processing, and more. 

### References:

* Di Khalil Ahmad, Abdullah. Wavelet Packets and Their Statistical Applications
