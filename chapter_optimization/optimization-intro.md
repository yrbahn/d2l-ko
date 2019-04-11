# Optimization and Deep Learning
# 최적화와 딥러닝

In this section, we will discuss the relationship between optimization and deep learning as well as the challenges of using optimization in deep learning. For a deep learning problem, we will usually define a loss function first. Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss. In optimization, a loss function is often referred to as the objective function of the optimization problem. Traditionally, optimization algorithms usually only consider minimizing the objective function. In fact, any maximization problem can be easily transformed into a minimization problem: we just need to use the opposite of the objective function as the new objective function.

이 섹션에서는 딥러닝에서 최적화를 사용하는 문제뿐만 아니라 최적화와 딥러닝 간의 관계에 대해 설명합니다. 딥러닝 문제에 대해, 우리는 일반적으로 손실 함수를 먼저 정의 할 것입니다. 손실 함수를 정의 한 후에는 그 손실을 최소화하기 위해 최적화 알고리즘을 사용할 수 있습니다. 최적화에서 손실 함수는 자주 최적화 문제의 목적 함수라고 언급합니다. 전통적으로 최적화 알고리즘은 대개 목적 함수를 최소화하는 것을 고려합니다. 사실, 모든 최대화 문제는 최소화 문제로 쉽게 변형 될 수 있습니다: 단지 그 목적 함수의 반대로 새로운 목적 함수로 사용해야합니다.


## The Relationship Between Optimization and Deep Learning
# 최적화와 딥러닝 사이의 관계

Although optimization provides a way to minimize the loss function for deep learning, in essence, the goals of optimization and deep learning are different.
In the ["Model Selection, Underfitting and Overfitting"](../chapter_deep-learning-basics/underfit-overfit.md) section, we discussed the difference between the training error and generalization error.
Because the objective function of the optimization algorithm is usually a loss function based on the training data set, the goal of optimization is to reduce the training error.
However, the goal of deep learning is to reduce the generalization error.
In order to reduce the generalization error, we need to pay attention to overfitting in addition to using the optimization algorithm to reduce the training error.

In this chapter, we are going to focus specifically on the performance of the optimization algorithm in minimizing the objective function, rather than the model's generalization error.

최적화는 딥러닝의 손실 함수를 최소화하는 방법을 제공하지만 근본적으로 최적화와 딥학습의 목표는 다릅니다.
[ "모델 선택, 언더 피팅 및 오버 피팅"](../ chapter_deep-learning-basics / underfit-overfit.md) 섹션에서 우리는 학습 오류와 일반화 오류의 차이점에 대해 논의했습니다.

최적화 알고리즘의 목적 함수는 대개 학습 데이터 셋을 기반으로 한 손실 함수이기 때문에 최적화의 목표는 학습 오류를 줄이는 것입니다.
그러나, 딥러닝의 목표는 일반화 오차를 줄이는 것이다.
일반화 오차를 줄이기 위해서는, 학습 오차를 줄이기 위해 최적화 알고리즘을 사용하는 것 외에도 오버 피팅에 주의 할 필요가 있다.

이 장에서는 모델의 일반화 오류가 아닌 목적 함수를 최소화하는 최적화 알고리즘의 성능에 특히 초점을 맞출 것입니다.


## Optimization Challenges in Deep Learning
## 딥러닝에서 최적화 과제

In the [Linear Regression](../chapter_deep-learning-basics/linear-regression.md) section, we differentiated between analytical solutions and numerical solutions in optimization problems. In deep learning, most objective functions are complicated. Therefore, many optimization problems do not have analytical solutions. Instead, we must use optimization algorithms based on the numerical method to find approximate solutions, which also known as numerical solutions. The optimization algorithms discussed here are all numerical method-based algorithms. In order to minimize the numerical solution of the objective function, we will reduce the value of the loss function as much as possible by using optimization algorithms to finitely update the model parameters.

There are many challenges in deep learning optimization. Two such challenges are discussed below: local minimums and saddle points. To better describe the problem, we first import the packages or modules required for the experiments in this section.


1013/5000
[선형 회귀 분석](../ chapter_deep-learning-basics / linear- regression.md) 섹션에서는 최적화 문제에서 분석적 접근방법(analytical solutions)과 수치적 접근방법(numerical solutions)을 구별했습니다. 딥학습에서 대부분의 목적 함수는 복잡합니다. 따라서 많은 최적화 문제에는 분석적 접근방법이 없습니다. 대신, 우리는 수치적 방법에 기반한 최적화 알고리즘을 사용하여 근사해를 찾아야 합니다, 이것 또한 수치적 접근 방법으로 알려져 있습니다. 여기서 설명한 최적화 알고리즘은 모두 수치 기반 알고리즘입니다. 목적 함수의 수치적 해를 최소화하기 위해 최적화 알고리즘을 사용하여 모델 매개 변수를 유한하게 갱신함으로써 가능한 한 손실 함수의 값을 줄입니다.

딥러닝 최적화에는 많은 어려움이 있습니다. 아래에서 두 가지 문제가 논의됩니다 : 로컬 미니멈(local minimus) 및 새들 포인트(saddle point). 문제를 더 잘 설명하기 위해 먼저 이 섹션에서 실험에 필요한 패키지 또는 모듈을 가져옵니다.


```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mpl_toolkits import mplot3d
import numpy as np
```

### Local Minimums
### 지역 최소

For the objective function $f(x)$, if the value of $f(x)$ at $x$ is smaller than the values of $f(x)$ at any other points in the vicinity of $x$, then $f(x)$ could be a local minimum. If the value of $f(x)$ at $x$ is the minimum of the objective function over the entire domain, then $f(x)$ is the global minimum.

For example, given the function

$$f(x) = x \cdot \text{cos}(\pi x), \qquad -1.0 \leq x \leq 2.0,$$

we can approximate the local minimum and global minimum of this function. Please note that the arrows in the figure only indicate the approximate positions.

목적 함수 $f(x)$ 에 대해, $f(x)$의 값이 $x$ 근처의 다른 모든 점에서 $f(x)$의 값보다 작으면 $f(x)$는 지역 최소값 일 수 있습니다. 어떤 $x$에서 $f(x)$의 값이 전체 도메인에 대한 목적 함수의 최소 값이면 $f(x)$는 전체 최소값입니다.

예를 들어, 아래의 함수가 주어지면 

$$ f(x) = x \cdot \text{cos}(\pi x), \qquad -1.0 \leq x \leq 2.0,$$

이 함수의 로컬 최소값과 전역 최소값을 근사 할 수 있습니다. 그림의 화살표는 대략적인 위치만 나타냅니다.



```{.python .input  n=2}
def f(x):
    return x * np.cos(np.pi * x)

d2l.set_figsize((4.5, 2.5))
x = np.arange(-1.0, 2.0, 0.1)
fig,  = d2l.plt.plot(x, f(x))
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                  arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)');
```

The objective function of the deep learning model may have several local optimums. When the numerical solution of an optimization problem is near the local optimum, the numerical solution obtained by the final iteration may only minimize the objective function locally, rather than globally, as the gradient of the objective function's solutions approaches or becomes zero.


딥러닝 모델의 목적 함수는 여러개의 지역 최적들을 가질 수 있습니다. 최적화 문제의 수치적 해가 지역 최적 근처에있을 때, 최종 반복에 의해 얻어진 수치 해는 목적 함수의 해의 그라디언트(gradient)가 0으로 접근하거나 0이 될때 목적 함수를 전체적으로가 아니라 지역적으로 최소화 할 수 있다.

### Saddle Points
### 새들 포인트

As we just mentioned, one possible explanation for a gradient that approaches or becomes zero is that the current solution is close to a local optimum. In fact, there is another possibility. The current solution could be near a saddle point. For example, given the function

$$f(x) = x^3,$$

we can find the position of the saddle point of this function.

방금 언급했듯이, 0에 접근하거나 0이 되는 그라디언트에 대한 한 가지 가능한 설명은 현재 해가 지역 최적 값에 가깝다는 것입니다. 사실, 또 다른 가능성이 있습니다. 현재 해가 새들 포인터(saddle point) 근처에 있을 수 있습니다. 예를 들어, 아래 함수가 주어지면,

$$ f(x) = x^3,$$

우리는 이 함수의 새들 포인트(saddle point) 위치를 찾을 수 있습니다.


```{.python .input  n=3}
x = np.arange(-2.0, 2.0, 0.1)
fig, = d2l.plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
                  arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)');
```

Now, we will use another example of a two-dimensional function, defined as follows:

$$f(x, y) = x^2 - y^2.$$

We can find the position of the saddle point of this function. Perhaps you have noticed that the function looks just like a saddle, and the saddle point happens to be the center point of the seat.

이제 다음과 같이 정의된 이차원 함수의 또 다른 예를 사용할 것이다.

$$f(x, y) = x^2 - y^2.$$

이 함수의 새들 포인터의 위치를 찾을 수 있습니다. 아마도 함수가 새들 포인터(saddle point)처럼 보이고 새들 포인트(saddle point)가 자리의 중심점 일 수 있습니다.

```{.python .input  n=4}
x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
ax.plot([0], [0], [0], 'rx')
ticks = [-1,  0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

In the figure above, the objective function's local minimum and local maximum can be found on the $x$ axis and $y$ axis respectively at the position of the saddle point.

We assume that the input of a function is a $k$-dimensional vector and its output is a scalar, so its Hessian matrix will have $k$ eigenvalues (see the ["Mathematical Foundation"](../chapter_appendix/math.md) section). The solution of the function could be a local minimum, a local maximum, or a saddle point at a position where the function gradient is zero:

* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all positive, we have a local minimum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all negative, we have a local maximum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are negative and positive, we have a saddle point for the function.

The random matrix theory tells us that, for a large Gaussian random matrix, the probability for any eigenvalue to be positive or negative is 0.5[1]. Thus, the probability of the first case above is $0.5^k$. Since, generally, the parameters of deep learning models are high-dimensional ($k$ is large), saddle points in the objective function are more commonly seen than local minimums.

In deep learning, it is difficult, but also not necessary, to find the global optimal solution of the objective function. In the subsequent sections of this chapter, we will introduce the optimization algorithms commonly used in deep learning one by one. These algorithms have trained some very effective deep learning models that have tackled practical problems.

위 그림에서 목적 함수의 지역 최소값과 최대 값은 새들 포인터(saddle point) 지점의 $x$ 축과 $y$ 축에서 각각 찾을 수 있습니다.

우리는 함수의 입력이 $k$ 차원 벡터이고 그 출력이 스칼라라고 가정하므로 헤센 행렬은 $k$ 고유 값을 가질 것입니다 ([ "수학 기초"] (../ chapter_appendix/math.md) 섹션를 보십시오). 함수의 해는 함수 그래디언트가 0 인 위치에서 지역 최소값, 지역 최대 값 또는 새들 포인트(saddle point)가 될 수 있습니다.

* 제로 그라디언트 위치에 있는 함수의 헤센 행렬의 고유 값이 모두 양수 일 때 우리는 함수에 대해 지역 최소값을 갖습니다.
* 제로 그라디언트 위치에서 함수의 헤센 행렬의 고유 값이 모두 음수 일 때 우리는 함수에 대해 최대 값을 갖습니다.
* 제로 그라디언트 위치에 있는 함수의 헤센 행렬의 고유 값이 음수이고 양수이면 함수에 대한 새들 포인트에 있습니다.

랜덤 행렬 이론은 큰 가우시안 랜덤 행렬에 대해 임의의 고유치가 양수 또는 음수가 될 확률이 0.5 [1]임을 알려줍니다. 따라서 위의 첫 번째 경우의 확률은 $0.5^k$입니다. 일반적으로 딥러닝 모델의 매개 변수는 고차원이므로 ($k$가 큼), 목적 함수의 새들 포인트(saddle point)가 지역 최소값보다 일반적으로 더 보인다.

깊은 학습에서 목적 함수의 전역적 최적 해를 찾는 것은 어렵지만 필요적이지는 않습니다. 이 장의 후속 섹션에서는 심화 학습에서 일반적으로 사용되는 최적화 알고리즘을 하나씩 소개 할 것입니다. 이 알고리즘은 실용적인 문제를 다루는 매우 효과적인 답러닝 모델을 학습 했습니다.

## Summary
## 요약

* Because the objective function of the optimization algorithm is usually a loss function based on the training data set, the goal of optimization is to reduce the training error.
* Since, generally, the parameters of deep learning models are high-dimensional, saddle points in the objective function are more commonly seen than local minimums.


* 최적화 알고리즘의 목적 함수는 일반적으로 학습 데이터 세트를 기반으로 한 손실 함수이기 때문에 최적화의 목표는 학습 오류를 줄이는 것입니다.
* 일반적으로 딥러닝 모델의 매개 변수는 고차원이기 때문에 목적 함수에서 새들 포인터(saddle point)는 지역 최소값보다 일반적으로 더 잘 나타납니다.

## Exercises

* What other challenges involved in deep learning optimization can you think of?
* 딥러닝 최적화와 관련된 다른 과제에는 무엇일까요?


## Reference

[1] Wigner, E. P. (1958). On the distribution of the roots of certain symmetric matrices. Annals of Mathematics, 325-327.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2371)

![](../img/qr_optimization-intro.svg)
