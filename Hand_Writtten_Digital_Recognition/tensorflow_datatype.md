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
