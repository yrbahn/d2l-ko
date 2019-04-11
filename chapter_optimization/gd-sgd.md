# Gradient Descent and Stochastic Gradient Descent

In this section, we are going to introduce the basic principles of gradient descent. Although it is not common for gradient descent to be used directly in deep learning, an understanding of gradients and the reason why the value of an objective function might decline when updating the independent variable along the opposite direction of the gradient is the foundation for future studies on optimization algorithms. Next, we are going to introduce stochastic gradient descent (SGD).

이 섹션에서는 그라데이션 디센트(gradient descent)의 기본 원리를 소개 할 것입니다. 그라디언트 디센트를 딥러닝에서 직접 사용하는 것이 일반적이지 않지만 그래디언트에 대한 이해와 그래디언트의 반대 방향으로 독립 변수를 업데이트 할 때 목적 함수의 값이 감소 할 수 있는 이유는 최적 알고리즘에서 향후 스터디를 위한 기초입니다 다음으로, 스토캐스틱 그래디언트 디센트(SGD)를 소개 할 예정입니다.


## Gradient Descent in One-Dimensional Space

Here, we will use a simple gradient descent in one-dimensional space as an example to explain why the gradient descent algorithm may reduce the value of the objective function. We assume that the input and output of the continuously differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$ are both scalars. Given $\epsilon$ with a small enough absolute value, according to the Taylor's expansion formula from the ["Mathematical basics"](../chapter_appendix/math.md) section, we get the following approximation:


여기에서는 그래디언트 디센트 알고리즘이 목적 함수의 값을 줄일 수있는 이유를 설명하기 위해서 예로써 일차원 공간에서 간단한 그래디언트 디센트를 사용할 것이다. 우리는 연속적으로 미분 할 수 있는 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$의 입력과 출력이 모두 스칼라라고 가정합니다. ["수학적 기본"](../chapter_appendix/math.md) 섹션의 Taylor 확장 공식에 따르면 절대 값이 충분히 작은 $\epsilon$을 주어지면 다음과 같은 근사값을 얻습니다.


$$f(x + \epsilon) \approx f(x) + \epsilon f'(x) .$$

Here, $f'(x)$ is the gradient of function $f$ at $x$. The gradient of a one-dimensional function is a scalar, also known as a derivative.

Next, find a constant

여기서 $f'(x)$는 $x$에서 함수 $f$의 그래디언트 입니다. 알차원 함수의 그래디언트는 derivative라고도하는 스칼라입니다.

다음으로,

$\eta>0$,
to make $\left|\eta f'(x)\right|$ sufficiently small
so that we can replace $\epsilon$ with
$-\eta f'(x)$
and get

$\left|\eta f'(x)\right|$를 충분히 작게 만들기 위해 상수를 찾습니다.
그래서 우리는 $\epsilon$을 다음과 같이 대체 할 수 있습니다.
$-\eta f'(x)$
그리고 

$$f(x - \eta f'(x)) \approx f(x) -  \eta f'(x)^2.$$
을 얻습니다.


If the derivative $f'(x) \neq 0$, then $\eta f'(x)^2>0$, so

$$f(x - \eta f'(x)) \lesssim f(x).$$

This means that, if we use

$$x \leftarrow x - \eta f'(x)$$

to iterate $x$, the value of function $f(x)$ might decline. Therefore, in the gradient descent, we first choose an initial value $x$ and a constant $\eta > 0$ and then use them to continuously iterate $x$ until the stop condition is reached, for example, when the value of $f'(x)^2$ is small enough or the number of iterations has reached a certain value.


515/5000
미분 $ f '(x) \ neq 0 $, $ \ eta f'(x) ^ 2> 0 $이면

$$ f (x - \ eta f '(x)) \ lesssim f (x). $$

즉, 우리가

$$ x \ leftarrow x - \ η f '(x) $$

$ x $를 반복하려면 함수 $ f (x) $의 값이 감소 할 수 있습니다. 따라서 그라디언트 디센트에서 우리는 먼저 초기 값 $ x $와 상수 $ \ eta> 0 $를 선택한 다음 중지 조건에 도달 할 때까지 $ x $를 반복적으로 반복합니다 (예 : $ f '(x) ^ 2 $가 충분히 작거나 반복 횟수가 특정 값에 도달했습니다.


Now we will use the objective function $f(x)=x^2$ as an example to see how gradient descent is implemented. Although we know that $x=0$ is the solution to minimize $f(x)$, here we still use this simple function to observe how $x$ is iterated. First, import the packages or modules required for the experiment in this section.

이제 목적 함수 $ f (x) = x ^ 2 $를 예제로 사용하여 그래디언트 디센트가 구현되는 방법을 살펴 보겠습니다. $ x = 0 $이 $ f (x) $를 최소화하는 해답이라는 것을 알지만,이 간단한 함수를 사용하여 $ x $이 반복되는 방식을 관찰합니다. 먼저이 섹션에서 실험에 필요한 패키지 또는 모듈을 가져옵니다.



```{.python .input  n=3}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
import math
from mxnet import nd
import numpy as np
```

Next, we use $x=10$ as the initial value and assume $\eta=0.2$. Using gradient descent to iterate $x$ 10 times, we can see that, eventually, the value of $x$ approaches the optimal solution.


다음으로 초기 값으로 $ x = 10 $을 사용하고 $ \ eta = 0.2 $로 가정합니다. 그라디언트 디센트를 사용하여 $ x $ 10 번 반복하면 결국 $ x $의 값이 최적의 솔루션에 접근한다는 것을 알 수 있습니다.


```{.python .input  n=4}
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x  # f(x) = x* the derivative of x is f'(x) = 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results

res = gd(0.2)
```

The iterative trajectory of the independent variable $x$ is plotted as follows.


80/5000
독립 변수 $ x $의 반복 궤도는 다음과 같이 표시됩니다.


```{.python .input  n=5}
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])
    d2l.plt.plot(res, [x * x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')

show_trace(res)
```

## Learning Rate

The positive $\eta$ in the above gradient descent algorithm is usually called the learning rate. This is a hyper-parameter and needs to be set manually. If we use a learning rate that is too small, it will cause $x$ to update at a very slow speed, requiring more iterations to get a better solution. Here, we have the iterative trajectory of the independent variable $x$ with the learning rate $\eta=0.05$. As we can see, after iterating 10 times when the learning rate is too small, there is still a large deviation between the final value of $x$ and the optimal solution.


573/5000
위의 그래디언트 디센트 알고리즘에서 긍정적 인 $ \ eta $는 일반적으로 학습 속도라고합니다. 이것은 하이퍼 매개 변수이며 수동으로 설정해야합니다. 너무 작은 학습 속도를 사용하면 $ x $가 매우 느린 속도로 업데이트되므로 더 나은 솔루션을 얻으려면 더 많은 반복이 필요합니다. 여기에서 우리는 학습 속도 $ \ eta = 0.05 $를 갖는 독립 변수 $ x $의 반복적 인 궤도를가집니다. 우리가 볼 수 있듯이, 학습 속도가 너무 느릴 때 10 번 반복 한 후에도 $ x $의 최종 값과 최적의 솔루션 사이에는 여전히 큰 편차가 있습니다.

```{.python .input  n=6}
show_trace(gd(0.05))
```

If we use an excessively high learning rate, $\left|\eta f'(x)\right|$ might be too large for the first-order Taylor expansion formula mentioned above to hold. In this case, we cannot guarantee that the iteration of $x$ will be able to lower the value of $f(x)$. For example, when we set the learning rate to $\eta=1.1$, $x$ overshoots the optimal solution $x=0$ and gradually diverges.


387/5000
지나치게 높은 학습률을 사용하면 위에서 언급 한 1 차 테일러 확장 공식이 유지하기에는 $ \ left | \ eta f '(x) \ right | $가 너무 클 수 있습니다. 이 경우, $ x $의 반복이 $ f (x) $의 값을 낮출 수 있다고 보장 할 수는 없습니다. 예를 들어 학습 률을 $ \ eta = 1.1 $로 설정하면 $ x $는 최적 해 $ x = 0 $을 초과하여 점차적으로 빗나갑니다.


```{.python .input  n=7}
show_trace(gd(1.1))
```

## Gradient Descent in Multi-Dimensional Space

Now that we understand gradient descent in one-dimensional space, let us consider a more general case: the input of the objective function is a vector and the output is a scalar. We assume that the input of the target function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is the $d$-dimensional vector $\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$. The gradient of the objective function $f(\boldsymbol{x})$ with respect to $\boldsymbol{x}$ is a vector consisting of $d$ partial derivatives:


489/5000
우리가 1 차원 공간에서의 그래디언트 강하를 이해한다면보다 일반적인 경우를 생각해 봅시다 : 목적 함수의 입력은 벡터이고 출력은 스칼라입니다. 우리는 목표 함수 $ f : \ mathbb {R} ^ d \ rightarrow \ mathbb {R} $의 입력이 $ d $ 차원 벡터 $ \ boldsymbol {x} = [x_1, x_2, \ ldots, x_d] ^ \ top $. $ \ boldsymbol {x} $에 대한 목적 함수 $ f (\ boldsymbol {x}) $의 그래디언트는 $ d $ 편미분으로 구성된 벡터입니다 :



$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_d}\bigg]^\top.$$

For brevity, we use $\nabla f(\boldsymbol{x})$ instead of $\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$. Each partial derivative element $\partial f(\boldsymbol{x})/\partial x_i$ in the gradient indicates the rate of change of $f$ at $\boldsymbol{x}$ with respect to the input $x_i$. To measure the rate of change of $f$ in the direction of the unit vector $\boldsymbol{u}$ ($\|\boldsymbol{u}\|=1$), in multivariate calculus, the directional derivative of $f$ at $\boldsymbol{x}$ in the direction of $\boldsymbol{u}$ is defined as

간략하게하기 위해 우리는 $ \ nabla \ {\ boldsymbol {x}} f (\ boldsymbol {x}) $ 대신 $ \ nabla f (\ boldsymbol {x}) $를 사용합니다. 그라디언트의 각 부분 미분 요소 $ \ partial f (\ boldsymbol {x}) / \ partial x_i $는 입력 $ x_i $에 대한 $ f $ at $ \ boldsymbol {x} $의 변화율을 나타냅니다. 다 변수 미적분학에서 단위 벡터 $ \ boldsymbol {u} $ ($ \ | \ boldsymbol {u} \ | = 1 $)의 방향으로 $ f $의 변화율을 측정하기 위해 $ f $ \ boldsymbol {u} $ 방향의 $ at $ \ boldsymbol {x} $는 다음과 같이 정의됩니다.


$$\text{D}_{\boldsymbol{u}} f(\boldsymbol{x}) = \lim_{h \rightarrow 0}  \frac{f(\boldsymbol{x} + h \boldsymbol{u}) - f(\boldsymbol{x})}{h}.$$

According to the property of directional derivatives \[1，Chapter 14.6 Theorem 3\], the aforementioned directional derivative can be rewritten as


146/5000
방향성 유도체 \ [1, Chapter 14.6 정리 3 \]의 성질에 따라, 앞서 언급 한 방향 미분은 다음과 같이 재 작성 될 수있다.


$$\text{D}_{\boldsymbol{u}} f(\boldsymbol{x}) = \nabla f(\boldsymbol{x}) \cdot \boldsymbol{u}.$$

The directional derivative $\text{D}_{\boldsymbol{u}} f(\boldsymbol{x})$ gives all the possible rates of change for $f$ along $\boldsymbol{x}$. In order to minimize $f$, we hope to find the direction the will allow us to reduce $f$ in the fastest way. Therefore, we can use the unit vector $\boldsymbol{u}$ to minimize the directional derivative $\text{D}_{\boldsymbol{u}} f(\boldsymbol{x})$.

For $\text{D}_{\boldsymbol{u}} f(\boldsymbol{x}) = \|\nabla f(\boldsymbol{x})\| \cdot \|\boldsymbol{u}\|  \cdot \text{cos} (\theta) = \|\nabla f(\boldsymbol{x})\|  \cdot \text{cos} (\theta)$,
Here, $\theta$ is the angle between the gradient $\nabla f(\boldsymbol{x})$ and the unit vector $\boldsymbol{u}$. When $\theta = \pi$, $\text{cos }(\theta)$ gives us the minimum value $-1$. So when $\boldsymbol{u}$ is in a direction that is opposite to the gradient direction $\nabla f(\boldsymbol{x})$, the direction derivative $\text{D}_{\boldsymbol{u}} f(\boldsymbol{x})$ is minimized. Therefore, we may continue to reduce the value of objective function $f$ by the gradient descent algorithm:

방향 미분 $ \ text {D} _ {\ boldsymbol {u}} f (\ boldsymbol {x}) $는 $ \ boldsymbol {x} $에 따라 $ f $에 대한 모든 가능한 변경 비율을 제공합니다. $ f $를 최소화하기 위해 우리는 $ f $를 가장 빨리 줄일 수있는 방향을 찾고자합니다. 따라서 단위 벡터 $ \ boldsymbol {u} $을 사용하여 방향성 미분 $ \ text {D} _ {\ boldsymbol {u}} f (\ boldsymbol {x}) $를 최소화 할 수 있습니다.

$ \ text {D} _ {\ boldsymbol {u}} f (\ boldsymbol {x}) = \ | \ nabla f (\ boldsymbol {x}) \ | \ cdot \ | \ boldsymbol {u} \ | \ cdot \ text {cos} (\ theta) = \ | \ nabla f (\ boldsymbol {x}) \ | \ cdot \ text {cos} (\ theta) $,
여기서 $ \ theta $는 그라데이션 $ \ nabla f (\ boldsymbol {x}) $와 단위 벡터 $ \ boldsymbol {u} $ 사이의 각도입니다. $ \ theta = \ pi $ 일 때, $ \ text {cos} (\ theta) $는 우리에게 최소값 $ -1 $을줍니다. 따라서 $ \ boldsymbol {u} $가 그라디언트 방향 $ \ nabla f (\ boldsymbol {x}) $와 반대 방향 인 경우 방향 미분 $ \ text {D} _ {\ boldsymbol {u}} f (\ boldsymbol {x}) $가 최소화됩니다. 따라서 그라디언트 디센트 알고리즘을 사용하여 목적 함수 $ f $의 값을 계속 줄일 수 있습니다.


$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f(\boldsymbol{x}).$

Similarly, $\eta$ (positive) is called the learning rate.

Now we are going to construct an objective function $f(\boldsymbol{x})=x_1^2+2x_2^2$ with a two-dimensional vector $\boldsymbol{x} = [x_1, x_2]^\top$ as input and a scalar as the output. So we have the gradient $\nabla f(\boldsymbol{x}) = [2x_1, 4x_2]^\top$. We will observe the iterative trajectory of independent variable $\boldsymbol{x}$ by gradient descent from the initial position $[-5,-2]$. First, we are going to define two helper functions. The first helper uses the given independent variable update function to iterate independent variable $\boldsymbol{x}$ a total of 20 times from the initial position $[-5,-2]$. The second helper will visualize the iterative trajectory of independent variable $\boldsymbol{x}$.


724/5000
이제 2 차원 벡터 $ \ boldsymbol {x} = [x_1, x_2] ^ \ top $ 인 목적 함수 $ f (\ boldsymbol {x}) = x_1 ^ 2 + 2x_2 ^ 2 $를 입력 및 출력으로 스칼라. 그래서 우리는 그라데이션 $ \ nabla f (\ boldsymbol {x}) = [2x_1, 4x_2] ^ \ top $를 갖습니다. 우리는 초기 위치 $ [- 5, -2] $로부터의 기울기 하강에 의한 독립 변수 $ \ boldsymbol {x} $의 반복 궤적을 관찰 할 것입니다. 먼저 두 가지 헬퍼 함수를 정의 할 것입니다. 첫 번째 도우미는 주어진 독립 변수 업데이트 함수를 사용하여 초기 위치 $ [- 5, -2] $에서 독립 변수 $ \ boldsymbol {x} $를 총 20 회 반복합니다. 두 번째 도우미는 독립 변수 $ \ boldsymbol {x} $의 반복 궤적을 시각화합니다.


```{.python .input  n=10}
# This function is saved in the d2l package for future use
def train_2d(trainer):
    # s1 and s2 are states of the independent variable and will be used later
    # in the chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

# This function is saved in the d2l package for future use
def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Next, we observe the iterative trajectory of the independent variable at learning rate $0.1$. After iterating the independent variable $\boldsymbol{x}$ 20 times using gradient descent, we can see that. eventually, the value of $\boldsymbol{x}$ approaches the optimal solution $[0, 0]$.


287/5000
다음으로 우리는 학습 속도 $ 0.1 $에서 독립 변수의 반복적 인 궤적을 관찰한다. 독립 변수 $ \ boldsymbol {x}을 그라디언트 디센트를 사용하여 $ 20 번 반복 한 결과를 볼 수 있습니다. 결국 $ \ boldsymbol {x} $의 값은 최적의 해 [$ 0, 0] $에 접근합니다.


```{.python .input  n=15}
eta = 0.1

def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

show_trace_2d(f_2d, train_2d(gd_2d))
```

## Stochastic Gradient Descent (SGD)

In deep learning, the objective function is usually the average of the loss functions for each example in the training data set. We assume that $f_i(\boldsymbol{x})$ is the loss function of the training data instance with $n$ examples, an index of $i$, and parameter vector of $\boldsymbol{x}$, then we have the objective function

332/5000
심층 학습에서 목적 함수는 일반적으로 학습 데이터 집합의 각 예에 대한 손실 함수의 평균입니다. $ f_i (\ boldsymbol {x}) $는 $ n $ examples, $ i $의 인덱스 및 $ \ boldsymbol {x} $의 매개 변수 벡터가있는 학습 데이터 인스턴스의 손실 함수라고 가정합니다. 목적 함수



$$f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\boldsymbol{x}).$$

The gradient of the objective function at $\boldsymbol{x}$ is computed as

$ \ boldsymbol {x} $에서 목적 함수의 그래디언트는 다음과 같이 계산됩니다.


$$\nabla f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}).$$

If gradient descent is used, the computing cost for each independent variable iteration is $\mathcal{O}(n)$, which grows linearly with $n$. Therefore, when the model training data instance is large, the cost of gradient descent for each iteration will be very high.

Stochastic gradient descent (SGD) reduces computational cost at each iteration. At each iteration of stochastic gradient descent, we uniformly sample an index $i\in\{1,\ldots,n\}$ for data instances at random, and compute the gradient $\nabla f_i(\boldsymbol{x})$ to update $\boldsymbol{x}$:

561/5000
그래디언트 디센트를 사용하면 각 독립 변수 반복에 대한 컴퓨팅 비용은 $ \ mathcal {O} (n) $이며 $ n $에 따라 선형 적으로 증가합니다. 따라서 모델 트레이닝 데이터 인스턴스가 클 경우 각 반복에 대한 그래디언트 디센트 비용이 매우 높습니다.

SGD (Stochastic gradient descent)는 각 반복에서 계산 비용을 줄입니다. 확률 적 그라디언트 디센트의 반복마다 무작위로 데이터 인스턴스에 대한 인덱스 $ i \ in \ {1, \ ldots, n \} $을 균일하게 샘플링하고 그라데이션 $ \ nabla f_i (\ boldsymbol {x}) $ $ \ boldsymbol {x} $ 업데이트 :


$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_i(\boldsymbol{x}).$$

Here, $\eta$ is the learning rate. We can see that the computing cost for each iteration drops from $\mathcal{O}(n)$ of the gradient descent to the constant $\mathcal{O}(1)$. We should mention that the stochastic gradient $\nabla f_i(\boldsymbol{x})$ is the unbiased estimate of gradient $\nabla f(\boldsymbol{x})$.

여기에서 $ \ eta $는 학습 속도입니다. 각 반복에 대한 컴퓨팅 비용은 그라디언트 디센트의 $ \ mathcal {O} (n) $에서 상수 $ \ mathcal {O} (1) $로 떨어집니다. 확률 적 경사도 $ \ nabla f_i (\ boldsymbol {x}) $는 그라데이션 $ \ nabla f (\ boldsymbol {x}) $의 불편 추정치입니다.


$$\mathbb{E}_i \nabla f_i(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}) = \nabla f(\boldsymbol{x}).$$

This means that, on average, the stochastic gradient is a good estimate of the gradient.

Now, we will compare it to gradient descent by adding random noise with a mean of 0 to the gradient to simulate a SGD.


210/5000
이것은 평균적으로 확률 적 그라디언트가 그라데이션의 좋은 추정치라는 것을 의미합니다.

이제 SGD를 시뮬레이트하기 위해 그래디언트 디센트에 평균 0의 랜덤 노이즈를 그라데이션에 추가하여 비교해 보겠습니다.


```{.python .input  n=17}
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

show_trace_2d(f_2d, train_2d(sgd_2d))
```

As we can see, the iterative trajectory of the independent variable in the SGD is more tortuous than in the gradient descent. This is due to the noise added in the experiment, which reduced the accuracy of the simulated stochastic gradient. In practice, such noise usually comes from individual examples in the training data set.

SGD에서의 독립 변수의 반복적 인 궤적은 기울기 강하에서보다 더 험합니다. 이것은 시뮬레이션에서 추가 된 노이즈로 인해 시뮬레이션 된 확률 적 그라디언트의 정확성이 감소했기 때문입니다. 실제로 이러한 소음은 교육 데이터 세트의 개별 사례에서 비롯됩니다.


## Summary

* If we use a more suitable learning rate and update the independent variable in the opposite direction of the gradient, the value of the objective function might be reduced. Gradient descent repeats this update process until a solution that meets the requirements is obtained.
* Problems occur when the learning rate is tool small or too large. A suitable learning rate is usually found only after multiple experiments.
* When there are more examples in the training data set, it costs more to compute each iteration for gradient descent, so SGD is preferred in these cases.


577/5000
*보다 적절한 학습률을 사용하고 독립 변수를 그레디언트의 반대 방향으로 업데이트하면 목적 함수의 값이 감소 될 수 있습니다. Gradient descent는 요구 사항을 만족하는 솔루션을 얻을 때까지이 업데이트 프로세스를 반복합니다.
* 학습 속도가 작거나 너무 커서 문제가 발생합니다. 적절한 학습 속도는 대개 여러 번의 실험 후에 만 발견됩니다.
* 훈련 데이터 세트에 더 많은 예제가있는 경우 그래디언트 디센트에 대한 각 반복을 계산하는 데 더 많은 비용이 소요되므로 SGD가 이러한 경우에 선호됩니다.

## Exercises

* Using a different objective function, observe the iterative trajectory of the independent variable in gradient descent and the SGD.
* In the experiment for gradient descent in two-dimensional space, try to use different learning rates to observe and analyze the experimental phenomena.

* 다른 목적 함수를 사용하여 기울기 강하 및 SGD에서 독립 변수의 반복 궤적을 관찰합니다.
* 2 차원 공간에서의 그래디언트 디센트 실험에서 다른 학습 속도를 사용하여 실험 현상을 관찰하고 분석하십시오.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2372)

![](../img/qr_gd-sgd.svg)
