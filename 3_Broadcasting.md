# Broadcasting

两个不同shape相加时， a.shape=[4, 32, 8], b.shape=[8]  
在做 a+b 运算时， 计算机默认先将 b 扩展为 b.shape=[1, 1, 8], 然后扩展为 b.shape=[4, 32, 8] 
之后在做相加运算

可以节省空间  

注意：
> [4, 32, 14, 14] 和 [2, 32, 14, 14] 不能够做broadcast。这是因为第0个维度没有是1的。

```python
x = tf.random.normal([4, 32, 32, 3])

a = x + tf.random.normal([3])
# shape = [4, 32, 32, 3]

b = x + tf.random.normal([32, 32, 1])
# shape = [4, 32, 32, 3]

c = x + tf.random.normal([4, 1, 1, 1])
# shape = [4, 32, 32, 3]
```

```python
x = tf.random.normal([4, 32, 32, 3])

b = tf.broadcast_to(tf.random.normal([4, 1, 1, 1], [4, 32, 32, 3]))
shape = [4, 32, 32, 3]
```