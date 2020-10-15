[toc]

## <font color=red>**特征工程概述**</font>

假设现在我们有如下数据：
> 用户id，身高(height)，体重(weight)

要基于如上数据，判断一个人的身材好坏。显然，身高、体重中任意一个维度都不能直接得出一个人身材如何的结论。

在这个例子中，一个非常经典的构造特征是BMI指数：

![](https://latex.codecogs.com/svg.latex?BMI%20=%20\frac{weight}{height^2})

这样，通过BMI指数，就能帮助我们刻画一个人的身材如何。甚至我们可以丢弃原始的体重和身高数据。

特征工程，就是基于原有的维度特征X，创造新的特征X'。 基本的操作包括：衍生(升维)、筛选(降维)。

![特征工程概览.jpg](https://github.com/Canaan1216/FeatureManagement/image/pic1.png)

通过总结和归纳，通常认为特征工程包括以下方面：

![特征工程组成部分.jpg](https://github.com/Canaan1216/FeatureManagement/image/fm_component.jpg)

其中特征处理是特征工程的核心部分。

## <font color=red>**数据预处理**</font>

通过特征提取，我们能得到未经处理的特征，这时的特征可能有以下问题：
- **不属于同一量纲：** 即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。
- **信息冗余：** 对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。二值化可以解决这一问题。
- **定性特征不能直接使用：** 某些机器学习算法和模型只能接受定量特征的输入，那么需要将定性特征转换为定量特征。最简单的方式是为每一种定性值(枚兴值)指定一个定量值，但是这种方式过于灵活，增加了调参的工作。<br>通常使用哑编码的方式将定性特征转换为定量特征：假设有N种定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展特征赋值为1，其他扩展特征赋值为0。<br>哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用哑编码后的特征可达到非线性的效果。
- **存在缺失值：** 缺失值需要补充。

### <font color=purple>无量纲化</font>

常见的无量纲化方法有标准化和区间缩放法。标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0, 1]等。

#### 标准化

标准化需要计算特征的均值和标准差，标准化公式为：

![](https://latex.codecogs.com/svg.latex?x%27=\frac{x-\overline{X}}{S})

#### 区间缩放法

区间缩放法的思路有多种，常见的一种为利用两个最值进行缩放，公式表达为：

![](https://latex.codecogs.com/svg.latex?x%27=\frac{x-Min}{Max-Min})

对于次数、金额等长尾化严重的数据，可以考虑对top5%统一处理为1。

#### L2归一化

![](https://latex.codecogs.com/svg.latex?x%27=\frac{x}{\sqrt{\sum_j^m%20x_j^2}})

### <font color=purple>定量特征二值化</font>

定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0，公式表达如下：

![](https://latex.codecogs.com/svg.latex?x%27=\left\{\begin{matrix}1,\;x%3Ethreshold\\%200,\;x%20\leq%20threshold\end{matrix}\right.)

### <font color=purple>对定性特征哑编码</font>

例：对于province这一个字段，它共有34种可能的枚举取值："北京市"、"广东省"、"山东首"、...<br>

我们可以将其dummy化，转为"是否北京市"、"是否广东省"、"是否山东省"等34个字段。

## <font color=red>**特征选择**</font>

当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。

### <font color=purple>为什么要做特征选择</font>

一般说来，当固定一个分类器的话，所选择的特征数量和分类器的效果之间会满足如下曲线：特征数据在等于某个x(1≤x≤n)时达到最优。过多或过少都会使分类器的效果发生下降。

![特征数量与模型效果的关系.jpeg](https://github.com/Canaan1216/FeatureManagement/image/feature_cnt_result.jpeg)

#### 特征不足的影响

当特征不足时，极易发生数据重叠，这种情况下任何分类器都会失效。如下图所示，仅依赖X<sub>1</sub>或X~2~都是无法区分这两类数据的。

![特征过少.png](https://github.com/Canaan1216/FeatureManagement/image/too_little_features.png)

#### 特征冗余的影响

增加特征可以理解为向高维空间映射，当这个“维度”过高时，容易造成同类数据在空间中的距离边远，变稀疏，这也易使得很多分类算法失效。如下图所示，仅依赖x轴本可划分特征，但y轴的引入使得同一类别不再聚集。

![特征过多.png](https://github.com/Canaan1216/FeatureManagement/image/too_much_features.png)


通常来说，我们从两个方面考虑来选择特征：
- **特征是否发散：** 如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用；
- **特征与目标的相关性：** 与目标相关性高的特征，应当优选选择；

根据特征选择的形式又可以将特征选择方法分为3种(经典三刀)：

- Filter：过滤法，基于自变量和目标变量之间的关联情况，来选择特征。特征选择的过程与后序学习器无关。其评估手段是判断单维特征与目标变量之间的关系，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征；
- Wrapper：包装法，根据目标函数(通常是预测效果评分)，每次选择若干特征，或者排除若干特征；
- Embedded：集成法，学习器自身自动选择特征。先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣；

### <font color=purple>filter</font>

#### 方差选择法

使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。

#### 相关系数法

使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值。皮尔森(Pearson)相关系数公式：两个连续变量(X,Y)的pearson相关系数P~X,Y~等于它们之间的协方差cov(X,Y)除以它们各自标准差的乘积。

![](https://latex.codecogs.com/svg.latex?p_{X,Y}=\frac{cov(X,Y)}{\sigma_X%20\sigma_Y}=\frac{E((X-\mu_X)(Y-\mu_Y))}{\sigma_X%20\sigma_Y}=\frac{E((X-\mu_X)(Y-\mu_Y))}{\sqrt{E(X^2)-E^2(X)}\sqrt{E(Y^2)-E^2(Y)}})


#### 卡方检验

经典的卡方检验是检验定性自变量对定性因变量的相关性。假设自变量有N种取值，因变量有M种取值，考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量：

![](https://latex.codecogs.com/svg.latex?\chi%20^2%20=%20\sum%20\frac{(A-E)^2}{E})

不难发现，[这个统计量的含义简而言之就是自变量对因变量的相关性](https://wiki.mbalib.com/wiki/%E5%8D%A1%E6%96%B9%E6%A3%80%E9%AA%8C)。

#### 互信息法(MIC)

全称为：Mutual information and maximal information coefficient。经典的互信息也是评价定性自变量对定性因变量的相关性的，互信息计算公式如下：

![](https://latex.codecogs.com/svg.latex?I(X;Y)=\sum_{x%20\in%20X}\sum_{y%20\in%20Y}{p(x,y)log\frac{p(x,y)}{p(x)p(y)}})

为了处理定量数据，最大信息系数法被提出。<br>

### <font color=purple>Wrapper</font>

#### stability selection
稳定性选择法，是对基于L1正则方法的一种补充。基于L1正则方法的局限性在于：当面对一组关联的特征时，它往往只会选择其中的一项特征。为了减轻该影响，使用了随机化的技术，通过多次重新估计稀疏模型，用==特征被选择为重要的次数/总次数==来表征该特征最终的重要程度。

稳定性选择是一种基于抽样和选择相结合的方法，评估的方法可以是回归、SVM等可引入正则项的算法，理想情况下，重要特征的得分会接近100%，而无用的特征得分会接近于0。

#### recursive feature elimination(RFE)
递归特征消除的主要思想是反复构建模型，然后选出最好的(或者最差的)特征(根据系数来选)，把选出来的特征放到一边，然后在剩余的特征上重复这个过程，直到遍历了所有的特征。在这个过程中被消除的次序就是特征的排序。

RFE的稳定性很大程度上取决于迭代时，底层用的哪种模型。比如RFE采用的是普通的回归(LR)，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的。假如采用的是Lasso/Ridge，正则化的回归是稳定的，那么RFE就是稳定的。

RFECV通过交叉验证的方式执行RFE，以此来选择最佳数量的特征：对于一个数量为d的feature的集合，他的所有的子集的个数是2的d次方减1(包含空集)。指定一个外部的学习算法(比如SVM)，通过该算法计算所有子集的validation error。选择error最小的那个子集作为所挑选的特征。


### <font color=purple>Embedded</font>

feature importance有两种常用实现思路：

(1) mean decrease in node impurity: 

> feature importance is calculated by looking at the splits of each tree.<br>
The importance of the splitting variable is proportional to the improvement to the gini index given by that split and it is accumulated (for each variable) over all the trees in the forest.

就是计算每棵树的每个划分特征在划分准则(gini或者entropy)上的提升，然后聚合所有树得到特征权重

(2) mean decrease in accuracy:

> This method, proposed in the original paper, passes the OOB samples down the tree and records prediction accuracy.<br>
A variable is then selected and its values in the OOB samples are randomly permuted. OOB samples are passed down the tree and accuracy is computed again.<br>
A decrease in accuracy obtained by this permutation is averaged over all trees for each variable and it provides the importance of that variable (the higher the decreas the higher the importance).

简单来说，如果该特征非常的重要，那么稍微改变一点它的值，就会对模型造成很大的影响。可以对该维度的特征数据进行打乱，重新训练测试，打乱前的准确率减去打乱后的准确率就是该特征的重要度。该方法又叫permute。

sklearn中实现的是第一种方法。

#### Coefficients as Feature Importance
##### Linear Regression Feature Importance
线性回归模型算法得到的结果公式，是各特征的加权和。这些权重系数，可以直接当做一种特征重要性分数。

> **[coef_](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)：** array of shape (n_features) or (n_targets, n_features)

Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.


```python
from sklearn.linear_model import LinearRegression
import numpy as np
 
np.random.seed(0)
size = 5000
 
# A dataset with 3 features
X = np.random.normal(0, 1, (size, 3))
#Y = X0 + 2*X1 + noise
Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)
lr = LinearRegression()
lr.fit(X, Y)
 
# A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
 
print "Linear model:", pretty_print_linear(lr.coef_)

---
# 结果：
Linear model: 0.984 * X0 + 1.995 * X1 + -0.041 * X2
```

可以看出，模型很好地揭示出了数据背后隐含的结构信息。但假如我们想将权重当作特征重要度，则必须满足一个假设：特征与特征之间不存在相互关联(互相正交)。

如果一个数据集中，特征与特征之间存在相互关联，那模型将变得不稳定。非常小的数据扰动就会导致模型的很大变化，使得解释模型变得困难。

比如，我们有一个数据集，其真实模型形式为：Y=X<sub>1</sub>+X~2~，然后我们观测的数据为(ϵ为噪音)：

![](https://latex.codecogs.com/svg.latex?\hat{Y}=X_1%20+%20X_2%20+%20\epsilon)

我们假设X<sub>1</sub>与X<sub>2</sub>存在相关关系，比如X<sub>1</sub>~≈X<sub>2</sub>。取决于ϵ的大小，我们得到的模型的结果可能是：
- Y=2X<sub>1</sub>
- Y=-X<sub>1</sub>+3X<sub>2</sub>
- ...

对上面的案例，我们添加一点噪音：

```python
from sklearn.linear_model import LinearRegression
 
size = 100
np.random.seed(seed=5)
 
X_seed = np.random.normal(0, 1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X3 = X_seed + np.random.normal(0, .1, size)
  
Y = X1 + X2 + X3 + np.random.normal(0,1, size)
X = np.array([X1, X2, X3]).T
  
lr = LinearRegression()
lr.fit(X,Y)
print "Linear model:", pretty_print_linear(lr.coef_)

---
# 结果：
Linear model: -1.291 * X0 + 1.591 * X1 + 2.747 * X2
```
可以看到，系数之和达到了3，模型结果与之前相差甚远。从模型中看，X<sub>1</sub>对结果值起到负向作用，而X<sub>3</sub>起正向作用。可事实上，X<sub>1</sub>与X<sub>3</sub>的权重应该是接近的。

##### Regularized models

正则化是一种对模型添加约束或惩罚的方法，主要目的是为了防止模型过拟合，提升模型的泛化能力。我们的目标，由最小化损失函数*E(X,Y)*，变成最小化*E(X,Y)+α‖w‖*，其中w是特征的权重系数向量，||.||则通常指L1或L2正则化，α表示正则项权重。

> Q1：实现参数的稀疏有什么好处吗?<br><br>
一个好处是可以简化模型，避免过拟合。因为一个模型中真正重要的参数可能并不多，如果考虑所有的参数起作用，那么可以对训练数据预测得很好，但是对测试数据效果则会变得很差。另一个好处是参数变少可以使整个模型获得更好的可解释性。

<br>

> Q2：参数值越小代表模型越简单吗?<br><br>
是的。因为越复杂的模型，越是会尝试对所有的样本进行拟合，甚至包括一些异常样本点，这就容易造成在较小的区间里预测值产生较大的波动，这种较大的波动也反映了在这个区间里的导数很大，而只有较大的参数值才能产生较大的导数。因此复杂的模型，其参数值会比较大。

###### <font color=blue>L0范数</font>

L0是指向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。换句话说，让参数W是稀疏的。

但不幸的是，L0范数的最优化问题是一个NP hard问题，而且理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替。

###### <font color=blue>L1 regularization/Lasso</font>

L1正则化会添加如下惩罚项到损失函数中：

![](https://latex.codecogs.com/svg.latex?\alpha%20\sum_{i=1}^{n}|w_i|)

由于每个非零的系数都会加总到惩罚项中，上式会倾向于将比较弱的特征的系数置为0。因此L1正则化会产生稀疏权值矩阵，也就起到了特征筛选的作用。

###### <font color=blue>L2 regularization/Ridge regression</font>

L2正则化会添加如下惩罚项到损失函数中：

![](https://latex.codecogs.com/svg.latex?\alpha%20\sum_{i=1}^{n}w_i^2)

L2正则化倾向于使系数值平均分散。对于存在相关关系的特征，它们会得到相近的值。

对于上面的例子，假如我们有两个相似的目标函数：
- Y=1*X<sub>1<\sub>+1*X<sub>2<\sub>
- Y=2*X<sub>1<\sub>+0*X<sub>2<\sub>

如果用L1正则化，则惩罚项都是2α。但如果用L2正则化，则第一个模型的惩罚项是2α，第二个模型的惩罚项是4α。

这个特性会使得模型更稳定，小量的噪声数据不会造成特征系数的较大波动。示例如下：

```python
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import numpy as np

size = 100

# We run the method 10 times with different random seeds
for i in range(10):
    print "Random seed %s" % i
    np.random.seed(seed=i)
    X_seed = np.random.normal(0, 1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X3 = X_seed + np.random.normal(0, .1, size)
    Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
    X = np.array([X1, X2, X3]).T

    def pretty_print_linear(coefs, names=None, sort=False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst, key=lambda x: -np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name)
                          for coef, name in lst)

    lr = LinearRegression()
    lr.fit(X, Y)
    print "Linear model:", pretty_print_linear(lr.coef_)

    ridge = Ridge(alpha=10)
    ridge.fit(X, Y)
    print "Ridge model:", pretty_print_linear(ridge.coef_)
    print
```

```shell
# 运行结果：
Random seed 0
Linear model: 0.728 * X0 + 2.309 * X1 + -0.082 * X2
Ridge model: 0.938 * X0 + 1.059 * X1 + 0.877 * X2

Random seed 1
Linear model: 1.152 * X0 + 2.366 * X1 + -0.599 * X2
Ridge model: 0.984 * X0 + 1.068 * X1 + 0.759 * X2

Random seed 2
Linear model: 0.697 * X0 + 0.322 * X1 + 2.086 * X2
Ridge model: 0.972 * X0 + 0.943 * X1 + 1.085 * X2

Random seed 3
Linear model: 0.287 * X0 + 1.254 * X1 + 1.491 * X2
Ridge model: 0.919 * X0 + 1.005 * X1 + 1.033 * X2

Random seed 4
Linear model: 0.187 * X0 + 0.772 * X1 + 2.189 * X2
Ridge model: 0.964 * X0 + 0.982 * X1 + 1.098 * X2

Random seed 5
Linear model: -1.291 * X0 + 1.591 * X1 + 2.747 * X2
Ridge model: 0.758 * X0 + 1.011 * X1 + 1.139 * X2

Random seed 6
Linear model: 1.199 * X0 + -0.031 * X1 + 1.915 * X2
Ridge model: 1.016 * X0 + 0.89 * X1 + 1.091 * X2

Random seed 7
Linear model: 1.474 * X0 + 1.762 * X1 + -0.151 * X2
Ridge model: 1.018 * X0 + 1.039 * X1 + 0.901 * X2

Random seed 8
Linear model: 0.084 * X0 + 1.88 * X1 + 1.107 * X2
Ridge model: 0.907 * X0 + 1.071 * X1 + 1.008 * X2

Random seed 9
Linear model: 0.714 * X0 + 0.776 * X1 + 1.364 * X2
Ridge model: 0.896 * X0 + 0.903 * X1 + 0.98 * X2
```
如果只使用线性回归模型，特征系数会波动比较剧烈。加入L2正则项后，特征系数则较为稳定。<br><br>

> Q：为什么L1范数倾向于使特征系数变稀疏，而L2范数倾向于使特征系数变平滑?


**数学公式角度解释：**

![](https://latex.codecogs.com/svg.latex?L1=|w_1|+|w_1|+...+|w_n|,\frac{\partial%20L_1}{\partial%20w_i}=sign(w_i)=1\;or\;-1)
![](https://latex.codecogs.com/svg.latex?L2=\frac{1}{2}(w_1^2%20+%20w_2^2%20+...+%20w_n^2),\frac{\partial%20L_2}{\partial%20w_i}=w_i)

我们假设学习速率η为0.5：
- L1的权值更新公式，每次更新都固定减少一个特定值，那么经过若干次迭代之后，权值就有可能减少到0；
- L2的权值更新公式，权值每次都等于上一次的1/2，所以会收敛到较小的值但不为0；


##### Logistic Regression Feature Importance

我们也可以拟合一个逻辑回归模型([LR](http://note.youdao.com/s/6DZHuoN5))，使用相关系数作为特征重要系数(也即w~i~)。

> **[coef_](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)：** ndarray of shape (1, n_features) or (n_classes, n_features)<br>
Coefficient of the features in the decision function.

coef_ is of shape (1, n_features) when the given problem is binary. In particular, when multi_class='multinomial', coef_ corresponds to outcome 1 (True) and -coef_ corresponds to outcome 0 (False).

```
# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

![logistic_regression_fi.png](https://github.com/Canaan1216/FeatureManagement/image/Linear_Regression_FI.png)

#### Decision Tree Feature Importance

决策树算法，如[CART(分类回归树)](http://note.youdao.com/s/Ti6dJKZC)，基于Gini指数或信息熵计算特征重要分数。

##### CART Feature Importance

We can use the CART algorithm for feature importance implemented in scikit-learn as the DecisionTreeRegressor and DecisionTreeClassifier classes.

After being fit, the model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.

##### Random Forest Feature Importance

**Random Forest回归模型特征重要度**

> **[feature_importances_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)：** ndarray of shape (n_features)<br>
The values of this array sum to 1, unless all trees are single node trees consisting of only the root node, in which case it will be an array of zeros.

The higher, the more important the feature. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

```
# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

![rf_regression_fi.png](https://github.com/Canaan1216/FeatureManagement/image/RF_Regression_FI.png)

**Random Forest分类模型特征重要度**

> **[feature_importances_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)：** ndarray of shape (n_features)<br>
The values of this array sum to 1, unless all trees are single node trees consisting of only the root node, in which case it will be an array of zeros.

The higher, the more important the feature. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

```
# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```
![rf_classification_fi.png](https://github.com/Canaan1216/FeatureManagement/image/RF_Classification_FI.png)

**==More examples：==**

1、Mean decrease impurity

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Load boston housing dataset as an example

boston = load_boston()
# print boston["feature_names"]
print boston["DESCR"]

X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
rf.fit(X, Y)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True)
```

```
# ---result：---
Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:  
    :Number of Instances: 506 
    :Number of Attributes: 13 numeric/categorical predictive
    :Median Value (attribute 14) is usually the target
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None
    :Creator: Harrison, D. and Rubinfeld, D.L.
This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

...

Features sorted by their score:
[(0.4283, 'LSTAT'), (0.3527, 'RM'), (0.0722, 'DIS'), (0.0432, 'CRIM'), (0.0243, 'PTRATIO'), (0.0207, 'NOX'), (0.017, 'AGE'), (0.0158, 'TAX'), (0.0112, 'B'), (0.0073, 'INDUS'), (0.0062, 'RAD'), (0.0006, 'ZN'), (0.0005, 'CHAS')]
```

**2、Mean decrease accuracy**

```python
from sklearn.cross_validation import ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

import numpy as np

boston = load_boston()

X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor()
scores = defaultdict(list)

# crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc - shuff_acc) / acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True)
```


```
# ---result：---
Features sorted by their score:
[(0.7739, 'LSTAT'), (0.5568, 'RM'), (0.0899, 'DIS'), (0.0409, 'NOX'), (0.0377, 'CRIM'), (0.0198, 'PTRATIO'), (0.0163, 'TAX'), (0.0109, 'AGE'), (0.0053, 'B'), (0.0046, 'INDUS'), (0.0034, 'RAD'), (0.0006, 'CHAS'), (0.0001, 'ZN')]
```


##### XGBoost Feature Importance

This model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.

**XGBoost Regression Feature Importance**

```
# xgboost for feature importance on a regression problem
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = XGBRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

**XGBoost Classification Feature Importance**

```
# xgboost for feature importance on a classification problem
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

### <font color=purple>相关方法总结</font>

下面我们在同一个数据集上运行所有特征重要度评估方法，以比较各方法之间的异同。数据集来自[Friedman regression dataset](ftp://ftp.uic.edu/pub/depts/econ/hhstokes/e538/Friedman_mars_1991.pdf)，数据符合如下分布：

![](https://latex.codecogs.com/svg.latex?y=10sin(\pi%20x_1%20x_2)+20(x_3%20-0.5)^2%20+10x_4%20+5X_5%20+\epsilon)

其中，x<sub>1</sub>到x<sub>5</sub>符合[0,1)均匀分布，ϵ是符合N(0,1)分布的标准正态偏差。与此同时，x<sub>6</sub>至x<sub>10</sub>是噪声变量，且与目标变量相互独立。

我们会再添加x<sub>11</sub>到x<sub>14</sub>这4个变量，它们与x<sub>1</sub>到x<sub>4</sub>强相关(生成方式为：f(x)=x+N(0,0.01))。通过这种方式，会使x<sub>11</sub>、...、x<sub>14</sub>与x<sub>1</sub>、...、x<sub>4</sub>之间的相关系数达到0.999以上。

我们把所有方法得到的结果分值做归一化，以方便互相比较。对于RFE，top5特征的分值置为1，其余特征分值基于其排序结果置于0~1之间。

```python
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE

np.random.seed(0)

size = 750
X = np.random.uniform(0, 1, (size, 14))
# 生成750*14个[0，1)之间的随机数

Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] + np.random.normal(0, 1))
# Add 3 additional correlated variables (correlated with X1-X3)
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))

names = ["x%s" % i for i in range(1, 15)]

ranks = {}

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))

lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)

rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

# stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, Y)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)

rf = RandomForestRegressor()
rf.fit(X, Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

f, pval = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)

mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
    mine.compute_score(X[:, i], Y)
    m = mine.mic()
    mic_scores.append(m)

ranks["MIC"] = rank_to_dict(mic_scores, names)

r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print "\t%s" % "\t".join(methods)
for name in names:
    print "%s\t%s" % (name, "\t".join(map(str,[ranks[method][name] for method in methods])))
```


Feature | LIN.CORR | LINEAR REG | LASSO | MIC | RF | RFE | RIDGE | STABILITY | MEAN
---|---|---|---|---|---|---|---|---|---
x1 | 0.3 | ==1== | 0.79 | 0.39 | 0.18 | 1 | 0.77 | 0.61 | 0.63
x2 | 0.44 | 0.56 | 0.83 | 0.61 | 0.24 | 1 | 0.75 | 0.7 | 0.64
x3 | 0 | 0.5 | 0 | 0.34 | 0.01 | 1 | 0.05 | 0 | 0.24
x4 | ==1== | 0.57 | ==1== | ==1== | 0.45 | 1 | ==1== | ==1== | ==0.88==
x5 | 0.1 | 0.27 | 0.51 | 0.2 | 0.04 | 0.78 | 0.88 | 0.6 | 0.42
x6 | 0 | 0.02 | 0 | 0 | 0 | 0.44 | 0.05 | 0 | 0.06
x7 | 0.01 | 0 | 0 | 0.07 | 0 | 0 | 0.01 | 0 | 0.01
x8 | 0.02 | 0.03 | 0 | 0.05 | 0 | 0.56 | 0.09 | 0 | 0.09
x9 | 0.01 | 0 | 0 | 0.09 | 0 | 0.11 | 0 | 0 | 0.03
x10 | 0 | 0.01 | 0 | 0.04 | 0 | 0.33 | 0.01 | 0 | 0.05
x11 | 0.29 | 0.6 | 0 | 0.43 | 0.14 | 1 | 0.59 | 0.39 | 0.43
x12 | 0.44 | 0.14 | 0 | 0.71 | 0.12 | 0.67 | 0.68 | 0.42 | 0.4
x13 | 0 | 0.48 | 0 | 0.23 | 0.01 | 0.89 | 0.02 | 0 | 0.2
x14 | 0.99 | 0 | 0.16 | ==1== | ==1== | 0.22 | 0.95 | 0.53 | 0.61

**Linear correlation：**
- 由于每个feature都是单独评估的，所以x<sub>1</sub>、...、x<sub>4</sub>与x<sub>11</sub>、...、x<sub>14</sub>的分值是接近的；
- 因为噪音向量x<sub>5</sub>、...、x<sub>10</sub>与目标变量之间几乎不存在关联，所以它们的重要性系数都很小；
- 由于x<sub>3</sub>与目标变量之间是二次平方关系，而线性相关模型无法学习到这种关系，所以x<sub>3</sub>的重要性系数是0；

**Lasso：**
- 会将头部重要特征取出，而将其它特征的系数置为0；
- 在减少特征数量上，lasso无疑是有效的，但在数据可解释性上偏差(可能会误导我们x<sub>11</sub>、x~12~、x~13~是不重要的)；

**MIC：**
- 与相关系数法比较类似的是，MIC也会对所有feature"公平对待"；
- 此外，MIC能够找到x<sub>3</sub>与目标向量之间的非线性关系；

**RF：**
- 随机森林基于不纯度的排序方法，在头部几个feature之后，在特征重要度分数上会有一个"锐减"。从结果上看，第三大特征(x~2~，0.24)的分值，已经小于top1(x<sub>14</sub>)的4分之一了；

**Ridge regression：**
- 此方法会使各feature的权重相对均匀地分布，可见x<sub>11</sub>、...、x<sub>14</sub>与x<sub>1</sub>、...、x<sub>4</sub>的分值是接近的；

**Stability selection：**
- 这种方法能够兼顾数据可解释性与top特征提取。从示例中可以看到，它很好地识别了top头部特征(x<sub>1</sub>，x~2~，x<sub>4</sub>，x<sub>5</sub>)，同时它们的相关特征也得到了一个相对较高的分值；

## <font color=red>**特征降维**</font>

当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，训练时间长的问题，因此降低特征矩阵维度也是必不可少的。

常见的降维方法除了基于L1惩罚项的模型以外，另外还有主成分分析法(PCA)和线性判别分析(LDA)，线性判别分析本身也是一个分类模型。

PCA和LDA有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中，但是PCA和LDA的映射目标不一样：==PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能==。所以说<font color=purple>**PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法**</font>。


### <font color=purple>LDA</font>

LDA的全称是Linear Discriminant Analysis(线性判别分析)，是一种supervised learning。有些资料上也称为是Fisher’s Linear Discriminant。LDA的原理是，将带上标签的数据(点)，通过投影的方法，投影到维度更低的空间中，使得投影后的点，会形成按类别区分，一簇一簇的情况，相同类别的点，将会在投影后的空间中更接近。

我们首先来回顾一下线性分类器LDA：对于K-分类的一个分类问题，会有K个线性函数：

![](https://latex.codecogs.com/svg.latex?y_k%20(x)=w_k^T%20x+w_{k0})

上式实际上就是一种投影，是将一个高维的点投影到一条高维的直线上，LDA最求的目标是，给出一个标注了类别的数据集，投影到了一条直线之后，能够使得点尽量的按类别区分开，当k=2即二分类问题的时候，如下图所示：

![lda投影demo.gif](https://github.com/Canaan1216/FeatureManagement/image/LDA_touying_demo.gif)

下面我们来推导一下**二分类LDA**问题的公式。假设用来区分二分类的直线(投影函数)为：

![](https://latex.codecogs.com/svg.latex?y=w^T%20x)

LDA分类的目标是：使得不同类别之间的距离越远越好，同一类别之中的距离越近越好(投影后类内方差最小，类间方差最大)。

![lda思想.png](https://github.com/Canaan1216/FeatureManagement/image/LDA_mind.png)

所以我们需要定义几个关键的值：

- 类别i的原始中心点为(D~i~表示属于类别i的点)：

```math
m_i = \frac{1}{n_i}\sum_{x \in D_i}x
```

- 类别i投影后的中心点为：

```math
\widetilde{m_i}=w^T m_i
```

- 类别i投影后，类别点之间的方差为(我们认为投影之后y的值是不变的，比如1/0)：

```math
\widetilde{s_i}=\sum_{y \in Y_i}(y-\widetilde{m_i})^2
```
最终我们可以得到一个下面的公式，表示LDA投影到w后的损失函数(因为我们这里考虑的是二分类LDA，所以只有2个类别项)：

```math
J(w)=\frac{|\widetilde{m_1}-\widetilde{m_2}|^2}{\widetilde{s_1}^2 + \widetilde{s_2}^2}
```

分母表示每一个类别内的方差之和，分子表示两个类别中心点之间的距离平方，我们最大化J(w)就可以求出最优的w了。

现在J(w)公式里，w是不能被单独提出来的，我们需要想办法先将w单独提出来。

我们定义一个投影前的各类别分散程度的矩阵：如果某一个分类的点距离这个分类的中心点m~i~越近，则S~i~里面元素的值就越小；如果分类的点都紧紧地围绕着m~i~，则S~i~里面的元素值越更接近0。

```math
S_i =\sum_{x \in D_i}(x-m_i)(x-m_i)^T
```

那么J(w)的分母可以化为：

```math
\widetilde{s_i}=\sum_{x \in D_i}(w^T x- w^T m_i)^2=\sum_{x \in D_i}w^T(x-m_i)(x-m_i)^T w = w^T S_i w

\widetilde{s_1}^2 + \widetilde{s_2}^2 = w^T (S_1 + S_2) w = w^T S_w w
```
同样，J(w)的分子可以化为：

```math
|\widetilde{m_1}-\widetilde{m_2}|^2 = w^T (m_1 - m_2)(m_1 - m_2)^T w = w^T S_B w
```

这样，损失函数就可以化为如下的形式：

```math
J(w)=\frac{w^T S_B w}{w^T S_w w}
```

因为如果分子、分母都可以取任意值的话，会使得解的个数有无穷多个，因此**我们将分母限制为长度为1**(拉格朗日乘子法技巧)，并作为拉格朗日乘子法的限制条件，带入得到：

![](https://latex.codecogs.com/svg.latex?c(w)=w^T%20S_B%20w-\lambda%20(w^T%20S_w%20w%20-1)%20\\%20\Rightarrow%20\frac{dc}{dw}=2S_B%20w%20-%202\lambda%20S_w%20w%20=0\Rightarrow%20S_B%20w%20=%20\lambda%20S_w%20w)

如此，便转化为一个求特征值的问题。第i大的特征值，便对应w~i~。

### <font color=purple>PCA</font>

![pca1.png](https://github.com/Canaan1216/FeatureManagement/image/PCA1.png)

以上图为例，数据点大部分都分布在x2方向上，在x1方向上的取值近似相同，那么对于有些问题就可以直接将x1坐标的数值去掉，只取x2坐标的值即可。但是有些情况不能直接这样取，例如：

![pca2.png](https://github.com/Canaan1216/FeatureManagement/image/PCA2.png)

上图的数据分布在x1和x2方向都比较均匀，任一去掉一个坐标的数值可能对结果都会有很大的影响。这个时候就是PCA展现作用的时候了。黑色坐标系是原始坐标系，红色坐表系是我们后面构建的坐标系，如果我们的坐标系是红色的，那么这个问题就和上图的问题一致了，我们只需要去掉y2坐标系的数据即可。

假设我们有m个样本，每个样本有n维特征。现在我们要将特征维度降到k维，那么PCA的数学表达可以这样表示：

![](https://latex.codecogs.com/svg.latex?Z_{m%20\times%20k}%20=%20f(X_{m%20\times%20n}),k%3Cn)

在线性空间中，矩阵可以表示为一种映射，所以上面的问题可以转化为寻找这样一个==矩阵W== ，该矩阵可以实现上面的映射目的：

![](https://latex.codecogs.com/svg.latex?Z_{m%20\times%20k}%20=%20X_{m%20\times%20n}%20W_{n%20\times%20k})

在PCA中，数据从原来的坐标系转换到了新的坐标系。推导过程与LDA类似。

新坐标系的选择是由数据本身决定的。第一个新坐标轴选择的是原始数据中方差最大的方向，第二个新坐标轴的选择和第一个坐标轴正交且具有最大方差的方向。该过程一直重复，重复次数为原始数据中特征的数目。

会发现，大部分方差都包含在最前面的几个新坐标轴维度中。因此我们可以只选择前面几个坐标轴，即对数据进行了降维处理。 