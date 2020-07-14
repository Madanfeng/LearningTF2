# 维度变换
## `tf.reshape()`
```python
a = tf.random.normal([4, 28, 28, 3])
# shape = [batch, h, w, channels]

b = tf.reshape(a, [4, 784, 3])
# b = tf.reshape(a, [4, -1, 3])
# shape可以理解为[batch, pixels, channels]

c = tf.reshape(a, [4, 784*3])
# c = tf.reshape(a, [4, -1])
# shape 可以理解为 [batch, pixels_per_batch]
```
注意：  
需要保证reshape前后的 *batch\*h\*w\*channels* 需要一致

## `tf.tanspose`
```python
a = tf.random.nornal((4, 28, 28, 3))
# shape = [4, 28, 28, 3]

b = tf.transpose(a)
# shape = [3, 28, 28, 4]

c = tf.transpose(a, perm=[0, 3, 1, 2])
# shape = [4, 3, 28, 28]
```
总结：  
>1. `tf.tanspose` 默认把shape完全相反
>2. 通过 perm 参数可任意调换shape
>3. 通常tf中为[batch, h, w, channels], torch中为[batch, channels, h, w]

## `tf.expand_dims`
```python
a = tf.random.normal([4, 35, 8])

b = tf.expand_dims(a, axis=0)
# shape = [1, 4, 35, 8]

c = tf.expand_dims(a, axis=3)
# shape = [4, 35, 8, 1]

d = tf.expand_dims(a, axis=-1)
# shape = [4, 35, 8, 1]

e = tf.expand_dims(a, axis=-1)
# shape = [1, 4, 35, 8]
```

## `tf.squeeze`
注意： only squeeze for shape=1 dim
```python
tf.suqeeze(tf.zeros([1, 2, 1, 1, 3]))
# shape = [2, 3]

a = tf.zeros([1, 2, 1, 3])

b = tf.suqeeze(a, axis=0)
# shape = [2, 1, 3]

c = tf.suqeeze(a, axis=-2)
# shape = [1, 2, 3]
```
总结：  
> `tf.suqeeze` 默认会把所有shape=1的维度给删除

