# 资料搜索
- CSDN、知乎、b站是大家可以接触到的比较简单也比较好用的学习资源和平台，当然也可以活用AI，综合多方资料来获取信息
- 我们提倡每一位论文复现组的同学都应具备信息检索的能力，这一能力对我们后续学习一些挑战性概念时至关重要

- 做为论文复现组的成员，我们难免需要读到论文，除了arXiv之外，知网、IEEE、Web of Science这些也是优质的论文文献网站，但在这些平台上面查看论文需要收费，我们可以使用学校的账号登录，这样就不需要自己付费
- 我们可以在学校的融合门户平台 [https://info.jnu.edu.cn/](https://info.jnu.edu.cn/) -> 登录 -> 下拉至数字资源栏目 -> 选择文献网站进入
    - 知网在高校/机构搜索栏输入暨南大学选择后进入即可
    - IEEE进入默认登录学校账号
    - Web of Science 进去时会弹出身份验证首选项选择页面，选择机构 身份验证 (Shibboleth or Open Athens)即可

# 我们接下来的任务：看鱼书
- 在正式进入人工智能的相关学习前，我们需要先了解深度学习的一些基础概念，所以我们希望大家在下一次分享会前看完鱼书以下内容：
1.5（Numpy）、1.6（Matplotlib）、2章（感知机）、3章（神经网络）、4章（神经网络的学习）、7章（卷积神经网络）
这两部分很难，有能力的同学可以选择性看，我们不做要求：5章（误差反向传播法）、6章（与学习相关的技巧）
- 如果同学没有参加上一次的分享会，或者还对我们的学习目标感到模糊，强烈建议查看[上次分享会回顾笔记](https://papergroup.tech/markdown-viewer.html?file=%E6%8A%80%E6%9C%AF%E9%83%A8%E7%AC%AC%E4%B8%80%E6%AC%A1%E5%88%86%E4%BA%AB%E4%BC%9A)
- 在下一次分享会时，我们会请同学分享在这个过程中学习到的深度学习概念，同时带大家再次梳理一次深度学习的整体脉络
- 我们提倡每位同学看完鱼书后能写一份学习报告（不需要严格讲究格式），可以简单概述学习到的内容，也可以详细描述你学习的一切，让我们了解大家的进度以更好地针对性教学，上传至[论文复现提交处](https://f.wps.cn/g/ep5IgqHt/)，上传代号002
- [鱼书下载链接](https://usertshu.github.io/papersreplication.github.io/files/%E9%B1%BC%E4%B9%A6.pdf)

# Python测试题评讲环节
此份测试题用做学习与入门python，我们可以希望大家在学中做题，完成后可选择上传至提交处以获取我们的反馈，上传代号001。即使过了任务截止日期，我们依旧欢迎同学学习与练手~

## Level 1（基础级）

### 1. 文件夹里有多少图片？
（本题不做评讲，能够正常实现功能的代码即算正确）

**需求**：编写代码计算指定文件夹下的图片总数（仅统计 `png` 和 `jpg` 两种格式），需**确保子文件夹中的图片不被遗漏**。

### 2. 步入 OOP 的殿堂

**需求**：OOP(面向对象编程)是我们编程解决问题的一个重要思想。根据以下使用示例及输出内容，编写一个类 Dog：

```python
# 1. 初始化对象，传入名字
dogQ = Dog("Q")
# 2. 访问对象的 name 属性
print(dogQ.name)  # 输出：Q

# 3. 两个狗对象互动
dogX = Dog("X")
dogX.play_with(dogQ)  # 输出：X is playing with Q

# 4.  bark（叫）与 is_barking（是否在叫）状态管理
print(dogQ.is_barking())  # 初始状态：False
dogQ.bark()  # 输出：Q is barking!
print(dogQ.is_barking())  # 叫之后状态：True

# 5. get_food（喂食）后， bark 状态重置
dogQ.get_food()
print(dogQ.is_barking())  # 喂食后状态：False
```

**优秀同学代码分享**
```python
class Dog:
    # 初始化方法，当创建Dog对象时会被调用
    def __init__(self,name:str):
        self.name = name
        self.barking = False
    
    def is_barking(self):
        return self.barking
    
    def play_with(self,dog):
        print(f'{self.name} is playing with {dog.name}')
    
    def get_food(self):
        self.barking = False
    
    def bark(self):
        print(f'{self.name} is barking!') 
        self.barking = True
```

### 3. 二维向量类

**需求**：实现 `Vector2d` 类表示二维向量，支持以下核心功能：

- 向量加减法
- 求向量模长
- 求反向量
- 判断两向量是否垂直
- 判断两向量是否共线

**使用示例**（向量加法支持两种形式，实现任意一种即可）：

```python
# 初始化两个向量（x坐标，y坐标）
vec_a = Vector2d(3, 5)
vec_b = Vector2d(1, 2)

# 形式1：通过运算符重载实现加法
print(vec_a + vec_b)  # 输出：(4, 7)
# 形式2：通过成员方法 add 实现加法
print(vec_a.add(vec_b))  # 输出：(4, 7)
```

**优秀同学代码分享**
```python
# 导入 math 模块以使用 sqrt 函数
import math

class Vector2d:
    # 初始化二维向量
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    # 重载加法运算符 (+)，实现向量加法
    def __add__(self, other: 'Vector2d') -> 'Vector2d':
        return Vector2d(self.x + other.x, self.y + other.y)

    # 向量加法（成员方法版本）
    def add(self, other: 'Vector2d') -> 'Vector2d':
        return self.__add__(other)

    # 重载减法运算符 (-)，实现向量减法
    def __sub__(self, other: 'Vector2d') -> 'Vector2d':
        return Vector2d(self.x - other.x, self.y - other.y)

    # 向量减法（成员方法版本）
    def sub(self, other: 'Vector2d') -> 'Vector2d':
        return self.__sub__(other)

    # 计算向量的模长（L2范数）
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    # 计算反向量
    def negate(self) -> 'Vector2d':
        return Vector2d(-self.x, -self.y)

    # 判断两个向量是否垂直（点积为0）
    def is_perpendicular(self, other: 'Vector2d') -> bool:
        return self.x * other.x + self.y * other.y == 0

    # 判断两个向量是否共线（叉积为0）
    def is_collinear(self, other: 'Vector2d') -> bool:
        return self.x * other.y - self.y * other.x == 0

    # 自定义对象的字符串表示形式，用于print()
    def __str__(self):
        return f"({self.x}, {self.y})"
```

### 4. 电动车没电了！

**背景**：你骑着电动车出去跟朋友一起玩，然而路途遥远，返程时电动车 没 电 了！最后辛辛苦苦回到了家，为了避免这种情况再次发生，你决定要用计算机帮忙计算电动车能走的路程！

**需求**：实现 `EBicycle` 类，满足以下功能：

1. **初始化**：创建对象时传入 `总电量` 和 `最大速度` 两个参数。
2. **续航判断**：给定当前行驶速度 `v` 和目标路程 `s`，判断能否到达：
   - 若电量不足：输出 `无法以该速度到达目的地！`
   - 若电量充足：更新当前电量，并输出更新后的电量。
3. **充电功能**：调用充电方法后，当前电量重置为“总电量”。

**核心公式**（电动车续航与电量消耗计算）：

- 最大里程数：$S = \frac{v\eta C}{P + mv + nv^2}$
- 已知路程求消耗电量：可通过上述公式变形推导（$C$ 为当前电量）
- 固定参数：$\eta=0.7$（效率）、$P=2$（基础功率）、$m=0.1$（线性阻力系数）、$n=0.02$（二次阻力系数）

> tips：本题目的情景改编自论文复现组上任部长真实经历，他说推一个小时电动车真的很累！

**优秀同学代码分享**
```python
class EBicycle :
    def __init__(self, totalPower, maxSpeed):
        self.totalPower = totalPower
        self.maxSpeed = maxSpeed
        self.power = self.totalPower

    # 判断能否以速度 v 行驶路程 s
    def canIMakeIt(self,v,s) :
        # 最大里程
        maxS = (v * 0.7 * self.power)/(2 + 0.1 * v + 0.02 * v ** 2)
        
        # 目标路程小于等于最大里程
        if s <= maxS :
            consumed_power = s * (2 + 0.1 * v + 0.02 * v ** 2) / (v * 0.7)
            self.power -= consumed_power
            print(f"行驶后剩余电量: {self.power}")
        else :
            print("无法以该速度到达目的地！")

    def charge(self) :
        self.power = self.totalPower
```

## Level 2（进阶级）

### 1. 再看 OOP

**需求**：实现 `Math` 类，通过**静态方法**封装以下数学工具功能（可以无需实例化，直接通过类调用）：

- 求算术平方根（sqrt） -> 这可以直接pow或**，也能使用牛顿迭代法（更高效）、二分法实现
- 求绝对值（abs）
- 上取整（ceil）
- 下取整（floor） -> 注意需要考虑负数值得向下取整情况，不能直接int(x)
- 四舍五入取整（round）
- 幂运算（fastpow，要求使用快速幂算法实现，而不是使用编程语言自带或库中的的pow()函数，只需要实现正整数的快速幂运算即可）

**使用示例**（需严格保证以下调用格式可正确输出）：

```python
print(Math.sqrt(4))   # 输出：2.0（或2，允许浮点数微小误差）
print(Math.abs(-2))   # 输出：2
print(Math.fastpow(3, 10))  # 输出：59049
```

**说明**：平方根运算可能因浮点数精度产生微小误差，属于正常情况。

**快速幂算法说明**：

快速幂是一种巧妙的计算方法，可以更快地计算 $a^n$（比如 $3^{10}$）。

> 贴士：快速幂算法实际上涉及到二进制运算和算法时间复杂度等计算机科学的重要概念。为了让同学更容易理解，我们在下面用更通俗的方式来讲解这个算法的核心思想。

**核心思想**：不需要把 $a$ 连续乘 $n$ 次，而是利用"指数可以拆分"的特点来减少计算次数。

**举个例子**：计算 $3^{10}$

- 普通方法：$3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3$（需要做9次乘法）
- 快速幂方法：
  - 我们注意到 $10 = 8 + 2$
  - 所以 $3^{10} = 3^8 \times 3^2$
  - 而 $3^8 = (3^4)^2$，$3^4 = (3^2)^2$，$3^2 = 3 \times 3$
  - 这样只需要做4次乘法就能得到结果！

**实现思路**：

1. 从指数 $n$ 开始，如果 $n$ 是偶数，可以把 $a^n$ 转化为 $(a^2)^{n/2}$
2. 如果 $n$ 是奇数，先提取一个 $a$ 出来，变成 $a \times a^{n-1}$，然后继续处理 $a^{n-1}$
3. 重复上述过程，直到指数变为0

> 提示：判断一个数是奇数还是偶数，可以用 `n % 2` 来判断（余数为1则为奇数）

**快速幂示例代码**
```python
def ddpow(x,n): # 用迭代形式实现
    res = 1
    while n>0:
        if n & 1:
            res *= x
        x *= x
        n >>= 1
    return res

def dgpow(x,n): # 用递归形式实现
    if n==0:
        return 1
    
    if n & 1:
        half = dgpow(x,n//2)
        return half * half * x
    else:
        half = dgpow(x,n//2)
        return half * half
```

### 2. 阅读理解 - MLP计算

**背景**：MLP(多层感知机)是深度学习中最简单的一种神经网络，尝试实现 以下 MLP 模型：

![mlp](https://github.com/3030Scar/mdg_papergroup/blob/main/img/mlp_struct.png?raw=true)

> 图中使用圆形代表神经元，表示一个变量

**模型结构**（共两层映射）：

- 输入层（第1列）：3个神经元（记为 $x[1,1], x[1,2], x[1,3]$）
- 隐藏层（第2列）：4个神经元（记为 $x[2,1], x[2,2], x[2,3], x[2,4]$）
- 输出层（第3列）：2个神经元（记为 $x[3,1], x[3,2]$）

**计算规则**：
对于第 $a$ 列（$a \geq 2$）的第 $b$ 个神经元，其值为前一列（$a-1$ 列）所有神经元与对应权重的乘积之和：$x[a, b] = \sum_{c=1}^{m} x[a-1, c] \times w[a-1, c, b]$其中：

- $m$ 为第 $a-1$ 列的神经元个数（如计算 $x[2,b]$ 时，$m=3$；计算 $x[3,b]$ 时，$m=4$）
- $w[i, j, k]$ 表示第 $i$ 层（对应第 $i$ 列到第 $i+1$ 列）中，左侧第 $j$ 个神经元到右侧第 $k$ 个神经元的权重（共 $3×4 + 4×2 = 20$ 个权重）

**需求**：

1. 自由设定或者读取用户输入数据（$x[1,1], x[1,2], x[1,3]$）和所有权重 $w$（权重也可以通过随机数生成）。
2. 计算并输出输出层（第3列）两个神经元的最终结果。
3. 建议：将 MLP 封装为类实现（非强制，但推荐）。

> tips：题目对专业概念并没有硬性要求，主要考察通过阅读来理解计算方法并复现的能力。另外希望同学知道实际的MLP除了w（权重）还有b（偏置），但此处省略。

**优秀同学代码分享**
```python
class Layer:
    def __init__(self,n:int):
        """
        n：本层的神经元数量
        """
        self.neurons = [0 for i in range(n)] # 初始化该层所有神经元的值为0
        self.n = n # 记录神经元数量
        pass # pass主要用于占位，但此处是多余的，可以舍弃

    def __str__(self):  # 定义当打印Layer对象时输出的内容
        return str(self.neurons)
    
    def __repr__(self):  # 定义在交互式环境中直接输出Layer对象时显示的内容
        return self.__str__()

class MLP:
    def __init__(self,weights = None):
        """
        weights：权重，一个三维数组，其中weights[a][b][c] = w[a+1,b+1,c+1]
        """
        # 神经网络的层列表
        self.layers:list[Layer] = list() 
        # 权重列表
        self.weights:list[list[list[float]]] = list() if weights is None else weights
        pass

    def calculate(self,*inputs): # ** 使用打包/解包操作 **
        # 将输入值赋给输入层（第0层）的神经元
        self.layers[0].neurons = inputs
        # 遍历隐藏层和输出层，计算每个神经元的值
        # last_layer_index 是当前处理层的前一层的索引
        for last_layer_index,layer in enumerate(self.layers[1:]):
            last_layer = self.layers[last_layer_index]
            # 遍历当前层的每一个神经元
            for neuron_index in range(layer.n):
                res = 0
                # 计算加权和：前一层所有神经元的值与对应权重的乘积之和
                for last_neuron_index,last_neuron in enumerate(last_layer.neurons):
                    res += self.weights[last_layer_index][last_neuron_index][neuron_index] * last_neuron
                # 更新当前神经元的值
                layer.neurons[neuron_index] = res
        # 返回输出层（最后一层）
        return self.layers[-1]
    pass

# 此处同学代码有点小问题，我们帮助他进行了修正
def load_mlp_from_weights(weights:list[list[list[float]]]) -> MLP:
    mlp = MLP(weights)
    # 创建输入层
    num_input_neurons = len(weights[0])
    mlp.layers.append(Layer(num_input_neurons))
    # 隐藏层和输出层
    for weight_group in weights:
        num_dest_neurons = len(weight_group[0])
        mlp.layers.append(Layer(num_dest_neurons))
    return mlp

test_weights = [
    [
        [1,2,3,4],
        [5,6,7,8],
        [1,2,3,4],
    ],
    [
        [1,2],
        [4,3],
        [1,3],
        [2,4],
    ]
]

mlp = load_mlp_from_weights(test_weights)
print(mlp.layers)
print(mlp.calculate(2,3,4))
```

### 3. 爬楼梯

**题目来源**：LeetCode 70. 爬楼梯（[https://leetcode-cn.com/problems/climbing-stairs/](https://leetcode-cn.com/problems/climbing-stairs/)）

**需求**：输入一个整数 $n$（表示楼梯的台阶数），输出一个整数（表示爬到楼顶的不同方法数）。

**规则**：每次只能爬 1 级或 2 级台阶，且只剩 1 级台阶时，不能爬 2 级。

**要求**：需在 LeetCode 上通过所有测试用例后，再按提交要求上传代码。

**优秀同学代码分享**
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a,b = 1,2
        if n == 1: return 1
        if n == 2: return 2
        else:
            for i in range(n - 2): # 直接使用斐波那契递推法
                a,b = b,a+b
        return b
'''
实际上这就是一个f(n)=f(n-1)+f(n-2)的问题，我们只需要求出斐波那契数列前n项和即可解决问题
当然也有同学使用组合数学法及递归法（不建议）实现此问题，大家可以自行了解
'''
```

## Level 3（挑战级）

### 1. 数字华容道游戏
（本题不做评讲，能够正常实现功能的代码即算正确）

**背景**：数字华容道是经典滑块游戏，在 $N×N$ 方格中排列数字 1~N²-1（留有一个空白格），通过移动滑块使数字按顺序排列。

**示例**（4×4 华容道复原目标）：

混乱状态
[ 3] [ 8] [ 5] [ 2]
[ 1] [ 7][14][10]
[12][ 9][11]
[ 4][13] [ 6][15]
复原为
[ 1] [ 2] [ 3] [ 4]
[ 5] [ 6] [ 7] [ 8]
[ 9] [10] [11] [12]
[13] [14] [15]

**核心需求**：

1. 生成有解的 $N×N$ 华容道初始状态（建议从复原状态开始随机打乱，确保有解）。
2. 实现滑块移动逻辑（如通过输入方向键/指令移动空白格周边的数字）。
3. 无需制作 UI 界面，输出控制台格式的方格即可（参考下图）：
   ![参考示例图](https://github.com/3030Scar/mdg_papergroup/blob/main/img/%E7%BB%88%E7%AB%AF%E7%A4%BA%E4%BE%8B%E5%9B%BE.png?raw=true)

**扩展功能**（可选）：

- 用文件记录当前游戏的最少步数。
- 实现“输入移动方向后无需按回车”的即时操作。
- 使用诸如 `os.system('cls')` 的方式在每次更新棋盘前清屏。

### 2. 大学生勇闯洛谷

需在洛谷上通过所有测试用例后，再按提交要求上传代码。

#### 2.1洛谷 P4995 小跳蛙的跳跃

**题目链接**：[https://www.luogu.com.cn/problem/P4995](https://www.luogu.com.cn/problem/P4995)

**题目描述**：
小跳蛙要跳到每块石头上各一次（最终停在任意一块石头上），地面高度 $h_0=0$。从第 $i$ 块石头跳到第 $j$ 块石头的体力消耗为 $(h_i - h_j)^2$，从地面跳到第 $i$ 块石头的体力消耗为 $h_i^2$。要求计算消耗的**最大体力值**。

**输入格式**：

1. 第一行：正整数 $n$（石头个数）。
2. 第二行：$n$ 个正整数（分别表示第 1~n 块石头的高度 $h_i$，所有 $h_i$ 互不相同）。

**输出格式**：一行正整数（最大体力值）。

**优秀同学代码分享**
```python
n=int(input())
x=map(int,input().split(" "))
x=sorted(x)
xx=[]
# 贪心策略：每次都从当前剩余的石头中，交替取出最大和最小的高度，
# 放入路径中，以实现最大跳跃差。
while len(x)>1:
    xx+=[x[-1],x[0]]
    x=x[1:-1] # ** 删除列表头尾的方式很优雅 **
xx=[0]+xx+x
ans=0
for window in range(1,len(xx)):
    ans+=(xx[window]-xx[window-1])**2
print(ans)
```

#### 2.2洛谷 P1004 方格取数

**题目链接**：[https://www.luogu.com.cn/problem/P1004](https://www.luogu.com.cn/problem/P1004)（建议有信息竞赛基础的同学尝试）

**题目描述**：
在 $N×N$ 的方格图中（$N \leq 9$），部分方格有正整数，其余为 0。从左上角 A 点出发，只能向下或向右走，到达右下角 B 点，共走两次。走过的方格中的数会被取走（取走后变为 0），求两次取数的**最大总和**。
![示例图](https://github.com/3030Scar/mdg/blob/main/img/others1.png?raw=true)

**输入格式**：

1. 第一行：整数 $N$（方格图大小）。
2. 接下来若干行：每行三个整数（前两个为方格位置 $(x,y)$，第三个为该方格的数），一行单独的 `0 0 0` 表示输入结束。

**输出格式**：一行整数（两次取数的最大总和）。

**关于动态规划问题**：我们建议解决这类问题优先使用迭代法。虽然它与记忆化递归的时间复杂度相同，但迭代法避免了函数调用的额外开销和栈溢出的风险，通常在实际运行中效率更高。同学可以自行去了解时间复杂度的概念。
可以观看[这个视频](https://www.bilibili.com/video/BV1AB4y1w7eT/?share_source=copy_web&vd_source=2ce9574f7e97cdd1f936211a10045041)快速了解动态规划

**题解代码（使用迭代法）**
```python
from math import inf
import sys

def max_pick_sum(grid):
    n = len(grid) - 1              
    neg = -10**12
    steps = 2 * n - 1              
    dp = [[[neg]*(n+1) for _ in range(n+1)] for _ in range(steps)]
    dp[0][1][1] = grid[1][1]

    for k in range(1, steps):
        for x1 in range(1, n+1):
            y1 = k + 1 - (x1 - 1)
            if not (1 <= y1 <= n):
                continue
            for x2 in range(1, n+1):
                y2 = k + 1 - (x2 - 1)
                if not (1 <= y2 <= n):
                    continue
                best_prev = max(
                    dp[k-1][x1][x2],                       
                    dp[k-1][x1-1][x2] if x1 > 1 else neg,    
                    dp[k-1][x1][x2-1] if x2 > 1 else neg,  
                    dp[k-1][x1-1][x2-1] if x1 > 1 and x2 > 1 else neg
                )
                if best_prev == neg:
                    continue
                gain = grid[x1][y1]
                if x1 != x2 or y1 != y2:
                    gain += grid[x2][y2]
                dp[k][x1][x2] = max(dp[k][x1][x2], best_prev + gain)

    return dp[steps-1][n][n]

data = sys.stdin.read().strip().split()
if not data:
    return
it = iter(data)
n = int(next(it))
grid = [[0]*(n+1) for _ in range(n+1)]
for x, y, v in zip(it, it, it):
    x = int(x); y = int(y); v = int(v)
    if x == 0 and y == 0 and v == 0:
        break
    grid[x][y] = v
print(max_pick_sum(grid))
```

**参考代码（使用递归法）**
```python
def dfs(x1,y1,x2,y2):

    if x1 >= N or y1 >=N or x2 >= N or y2 >= N:
        return float('-inf')

    if x1 == N-1 and y1 == N-1 and x2 == N-1 and y2 == N-1:
        return grid[N-1][N-1]

    state = (x1,y1,x2,y2)

    if state in dp:
        return dp[state]

    crt_score = grid[x1][y1]
    if (x1,y1) != (x2,y2):
        crt_score += grid[x2][y2]

    max_future = max(
        dfs(x1+1 , y1 , x2+1 , y2),
        dfs(x1+1 , y1 , x2 , y2+1),
        dfs(x1 , y1+1 , x2+1 , y2),
        dfs(x1 , y1+1 , x2 , y2+1)
        )
    dp[state] = crt_score + max_future
    return dp[state]

N = int(input())
grid = [[0 for _ in range(N)] for _ in range(N)]
dp = {}
while True:
    x, y, value = map(int,input().split())
    if x == 0 and y == 0 and value == 0:
        break
    grid[x-1][y-1] = value

print(dfs(0,0,0,0))
```