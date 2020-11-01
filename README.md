# Optimization
We analyze some standard optimization techniques such Fixed Point method, Gradient Descent and Stochastic Gradient Descent.

## Solving Fixed Point Problems

In numerous applications, we encounter the task of solving equations of the form $$x = g(x)$$
for a continuous function $g$. In this exercise we will see one simple method to solve such problems: $$x_{t+1} = g(x_t)\,.$$
We will solve two equations of this form: $$x = log(1+x)$$ and $$x = log(2+x)\,.$$ We compare the convergence of for these two functions above.



## Gradient Descent

We perform gradient descent on [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) Least squares estimation is one of the fundamental machine learning algorithms. Given an $ n \times d $ matrix $A$ and a $ n \times 1$ vector $b$, the goal is to find a vector $x \in \mathbb{R}^d$ which minimizes the objective function $$f(x) = \frac{1}{2n} \sum_{i=1}^{n} (a_i^\top x - b_i)^2 = \frac{1}{2n} \|Ax - b\|^2 $$ 

We will try to fit $x$ using Least Squares Estimation. One can see the function is $L$ smooth with $L = \frac1n\|A\|^2$

We compare the converge assuming bounded gradients and constant step size.



### Assuming bounded gradients

Assume we are moving in a bounded region $\|x\| \leq 25$ containing all iterates (and we assume $\|x-x^\star\| \leq 25$ as well, for simplicity). Then by $\nabla f(x) = \frac{1}{n}A^\top (Ax - b)$, one can see that $f$ is Lipschitz over that bounded region, with Lipschitz constant $\|\nabla f(x)\| \leq \frac{1}{n} (\|A^\top A\|\|x\| + \|A^\top b\|)$



## Stochastic Gradient Descent

We perform SGD and stochastic SGD again on [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) to minimize the same objective function. We compare convergence assuming bounded expected gradients and strong convexity and smoothness

### Assuming bounded expected stochastic gradients

Assume we are moving in a bounded region $\|x\| \leq 25$ containing all iterates (and we assume $\|x-x^\star\| \leq 25$ as well, for simplicity). By $\nabla f(x) = \frac{1}{n}A^\top (Ax - b)$, one can see that $f$ is Lipschitz over that bounded region, with Lipschitz constant \nabla f(x) = \frac{1}{n}A^\top (Ax - b). We also know that $E\big[\|g_t\|\big | x_t\big]\ = \nabla f(x)$. So to find B such that  $E\big[\|g_t\|^2\big]\leq B^2$, we need to compute the Lipschitz constant.

### Strongly convex function

One can see the function is $\mu$ strongly convex with $\mu = \lambda_{max}(\nabla^2 f(x))$ and $L$ smooth with $L = \lambda_{min}(\nabla^2 f(x)$ everywhere, since here the Hessian matrix is constant, independent of $x$.
