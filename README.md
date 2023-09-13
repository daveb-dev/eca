# MCS: Efficient Computational Algorithms SA 2023-2024
Michael Multerer, Davide Baroli, Valentina Ghidini and Sara Avesani.

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

## 1. Title Topic  (Supervisor: XXX)
Text (10 lines)

### References:
(2 main references)
* B. Korte and J. Vygen. Combinatorial Optimization.

## 1. Approximate Bayesian Computation (ABC)  (Supervisor: Valentina)
In Bayesian statistics, the posterior distribution of a parameter $\theta$ given a dataset $D$ is usually computed using Bayes' rule
$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)} \propto p(D | \theta) p(\theta)$$
where $p(\theta)$ is the prior distribution of the parameter, $p(D)$ the marginal distribution of the data (which usually acts as a normalising constant, and can be omitted) and $p(D|\theta)$ is the likelihood. In many cases, the likelihood $p(D|\theta)$ is either unknown in closed form, or computationally intractable, but nevertheless necessary for posterior inference. Hence the need to resort to approximate computation. To this aim, there exist different classes of algorithms, among which we can find ABC-based ones.

### References:
* <a href="https://allendowney.github.io/ThinkBayes2/chap20.html"> Allen B. Downey, Think Bayes (2021) - Chapter 20 </a>
* <a href="https://pubmed.ncbi.nlm.nih.gov/23341757/"> Sunnaker et al., Approximate Bayesian computation (2019)</a>

## 2. Inverse Transform Sampling (Supervisor: Valentina)
In a plethora of computational problems, there is a need to simulate from a given distribution. For example, it may be necessary to simulate white noise by sampling from a standard gaussian distribution. But how does one do that? 
An elegant way is to use Inverse Transform Sampling to sample $X \sim F_X(x)$, where $F_X(x) = P(X \leq x)$ is the cumulative distribution function (CDF) of $X$. To use this algorithm, we only need to have $F_X(\cdot)$ invertible and and efficient way to sample from a uniform distribution on $[0,1]$ (spoiler: it exists!).

### References

* <a href="https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html"> Easy blogpost </a>
* Soon-to-be-found appropriate reference
## Programming Languages for code

<p align="center">
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/python_logo.png"></img>
  <img  height="300" src="https://github.com/EfficientComputationAlgorithm/eca/blob/main/images/cpp_logo.png"></img>
</p>



