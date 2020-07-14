# 数学运算
outline:  
> +, -, *, /
> **, square, pow
> sqrt
> //, %
> exp, log
> @, matmul
> linear layer

1.element-wise
> +, -, *, /  
2.matrix-wise
> @, matmul
3.dim-wise
> reduce_mean/max/min/sum

## + - * / % //
```python
a = tf.ones([2, 2])
b = tf.fill([2, 2], 2.)

c1 = a+b    # [[3., 3.], [3., 3.]]
c2 = a-b    # [[-1., -1.], [-1., -1.]]
c3 = a*b    # [[2., 2.], [2., 2.]]
c4 = a/b    # [[0.5, 0.5], [0.5, 0.5]]
c5 = b//a   # [[2., 2.], [2., 2.]]
c6 = b%a    # [[0., 0.], [0., 0.]]
```

## `tf.math.log` & `tf.exp`
```python
a = tf.ones([2, 2])

b = tf.math.log(a)    # [[0., 0.], [0., 0.]]
c = tf.exp(a)         # [[2.7182817, 2.7182817], [2.7182817,2.7182817]]
```

## `tf.pow` & `tf.sqrt`
```python
a = tf.fill([2, 2], 2.)

c1 = tf.pow(a, 3)    # [[8., 8.], [8., 8.]]
c2 = a ** 3          # [[8., 8.], [8., 8.]]
c3 = tf.sqrt(a)      # [[1.4142135, 1.4142135], [1.4142135, 1.4142135]]
```

## `@` & `rf.matmul`
```python
a = tf.ones([2, 2])
b = tf.fill([2, 2], 2.)

c1 = a @ b             # [[4., 4.], [4., 4.]]
c2 = tf.matmul(a, b)   # [[4., 4.], [4., 4.]]
```
