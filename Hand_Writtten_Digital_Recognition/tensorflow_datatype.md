# tensorflow datatype
## 1. Tensor
### 1.1 在CPU, GPU定义
```buildoutcfg
with tf.device(cpu):
    a = tf.constant([1.0, 2.0])
with tf.device(gpu):
    b = tf.constant([3.0, 4.0])

# 查看所在设备
print(a.device)
print(b.device)
```
### 1.2 CPU, GPU 相互转换
```buildoutcfg
a_gpu = a.gpu()
b_cpu = b.cpu()

print(a.device)
print(b.device)
```
>*注意：不同设别的数据不能相互运算！*
### 1.3 查看维度
```buildoutcfg
print(a.ndim)
print(a.rank)
```
### 1.4 判断是否为tensor
```buildoutcfg
print(isinstance(a, tf.Tensor))
print(tf.is_tensor(a))
```
*推荐使用`tf.is_tensor()`.*
### 1.5 数据类型的相互转换 `tf.cast`和 `tf.convert_to_tensor`
```buildoutcfg
c = np.arange(5)
print(c.dtype)

c_tensor = tf.conver_to_tensor(c)
print(c_tensor)

c_tensor = tf.cast(c_tensor, dtype=tf.double)  # 转化成其他类型
print(c_tensor.dtype)
```
以及比较特殊的数据类型`tf.Variable()`
```buildoutcfg
d = tf.range(5)
e = tf.Variable(d, name='input_data')
print(e.dtype)       # tf.int32
print(e.name)        # input_data:0
print(e.trainable)   # True

print(isinstance(b, tf.Tensor)    # False
print(isinstance(b, tf.Variable)  # True
print(tf.is_tensor(b))            # True  

print(b.numpy())
# array([0, 1, 2, 3, 4], dtype=int32)
```

## 2.创建Tensor
### 2.1 From Numpy, List
```buildoutcfg
print(tf.convert_to_tensor(np.ones([2, 3])))

print(tf.convert_to_tensor([1, 2]))
```
### 2.2 `tf.zeros` 和 `tf.zeros_like`
```buildoutcfg
print(tf.zeros([]))
# <tf.Tensor:id=2, shape=(), dtype=float32, numpy=0.0>

print(tf.zeros([1))
# <tf.Tensor:id=6, shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>

print(tf.zeros([2, 2])
# <tf.Tensor:id=10, shape=(2, 2), dtype=float32, numpy=array([[0., 0.], [0., 0.]], dtype=float32)>

print(tf.zeros([2, 3, 3]))
# <tf.Tensor:id=14, shape=(2, 3, 3), dtype=float32, numpy=array([[0., 0., 0.], ... [0., 0., 0.]], dtype=float32)>

a = tf.zeros([2, 3, 3])
print(tf.zeros_like(a))
# <tf.Tensor:id=19, shape=(2, 3, 3), dtype=float32, numpy=array([[0., 0., 0.], ... [0., 0., 0.]], dtype=float32)>
print(tf.zeros(a.shape))
# <tf.Tensor:id=23, shape=(2, 3, 3), dtype=float32, numpy=array([[0., 0., 0.], ... [0., 0., 0.]], dtype=float32)>
```
总结：
>1.`tf.zeros(.)` 括号中为shape。  
>2.`tf.zeros_like(.)` 括号中为tensor。  
>3.`tf.ones(.)` 和 `tf.ones_like(.)` 同上。
  
### 2.3 `tf.fill`  
```buildoutcfg
print(tf.fill([2, 2], 0))
# <tf.Tensor:id=91, shape=(2, 2), dtype=float32, numpy=array([[0., 0.], [0., 0.]], dtype=float32)>
```
总结：
>`tf.fill()` 中，需要传shape和value

### 2.4 Normal
```buildoutcfg
print(tf.random.normal([2, 2], mean=1, stddev=1))
# <tf.Tensor:id=57, shape=(2, 2), dtype=float32, numpy=array([[2.3931956, 1.0709286], [0.33781078, 1.1542176]], dtype=float32)>

print(tf.random.truncated_normal([2, 2], mean=0, stddev=1))
# <tf.Tensor:id=64, shape=(2, 2), dtype=float32, numpy=array([[-056123465, 1.3936464], [-1.1112062, -0.76954645]], dtype=float32)>

print(tf.random.uniform([2, 2], minval=0, maxval=1))
# <tf.Tensor:id=79, shape=(2, 2), dtype=float32, numpy=array([[0.1432991, 0.0267868], [0.08879011, 0.8807217]], dtype=float32)>
```
总结：
>1. `tf.random.normal()` 中默认是（0， 1）**_正态分布_**
>2. `tf.random.truncated_normal()` 为 **_截断的正态分布_**， 可适用于sigmod激活函数带来的 **_梯度消失_**
>3. `tf.random.uniform()` 为 **_均匀分布_**

### 2.5 `tf.constant`
注意： 每行必须有相同维度的元素

## 3. 索引和切片  
### Selective Indexing
例如，有班级的数据集，data:[classes, students, subjects]  # [4, 35, 8]
代表有classes个班级，每个班级students个学生，每个学生有subjects个科目。
#### 1.`tf.gather`
只在某一个维度
```buildoutcfg
tf.gather(data, axis=0, indices=[2, 1, 3, 0])
# axis=0, 表示取第0个维度，即班级。
# indices=[2, 1, 3, 0], 代表班级的次序为原来的index的第2， 1， 3， 0。
# shape = [4, 35, 8]

tf.gather(data, axis=1, indices=[2, 3, 7, 9, 16])
# 同理，对每个班级的第2，3，7，9，16个同学进行抽取所有科目。
# shape = [4, 5, 8]
```

#### 2. `tf.gather_nd`
多个维度
```buildoutcfg
tf.gather_nd(data, [0])
# 取0号班级的所有学生的所有科目
# shape = [35, 8]

tf.gather_nd(data, [0, 1])
# 取0号班级的1号学生的所有科目
# shape = [8]

tf.gather_nd(data, [0, 1, 2])
# 取0号班级的1号学生的2号科目
# shape = []

tf.gather_nd(data, [[0, 1, 2]])
# 取0号班级的1号学生的2号科目
# shape = [1]

tf.gather_nd(data, [[0, 1], [2, 3]])
# 取 0号班级的1号学生 和 2号班级的3号学生 的所有科目
# shape = [2, 8]

tf.gather_nd(data, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# 取 0号班级的0号同学的0号科目 和 1号班级的1号同学的1号科目 和 2号班级的2号同学的2号科目
# shape = [3]

tf.gather_nd(data, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]])
# 取 0号班级的0号同学的0号科目 和 1号班级的1号同学的1号科目 和 2号班级的2号同学的2号科目
# shape = [1, 3]
```

#### 3. `tf.boolean_mask`
```buildoutcfg
data.shape = [4, 28, 28, 3]

tf.boolean_mask(data, mask=[True, True, False, False])
# 取前两个图片， shape = [2, 28, 28, 3]

tf.boolean_mask(data, mask=[True, True, False], axis=3)
# 取所有图片的 R G 两个通道， shape = [4, 28, 28, 2]

a = tf.ones([2, 3, 4])
tf.boolean_mask(a, mask=[[True, False, False], [False, True, True]])
# 取 0号的第0个， 1号的第1，2个， 其中每个有4维
# shape = [3, 4]
```

