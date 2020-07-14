import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

BATCH_SIZE = 32


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_val, y_val) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(BATCH_SIZE).repeat(10)

db_val = tf.data.Dataset.from_tensor_slices((x_val, x_val))
db_val = db.map(preprocess).batch(BATCH_SIZE)

db_iter = iter(db)
sample = next(db_iter)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu)
])

model.build(input_shape=[None, 28*28])
model.summary()

optimizer = optimizers.Adam(lr=1e-3)

acc_meter  = metrics.Accuracy()
loss_meter = metrics.Mean()


def main():

    for epoch in range(10000):
        for step, (x, y) in enumerate(db):

            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape as tape:

                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

                loss_meter.update_state(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss_meter.result().numpy())
                loss_meter.reset_states()

            # evaluate
            if step % 100 == 0:
                total_correct = 0
                total_num     = 0
                acc_meter.reset_states()

                for step, (x, y) in enumerate(db_val):
                    x = tf.reshape(x, [-1, 28*28])
                    logits = model(x)
                    prob = tf.nn.softmax(logits, axis=1)
                    pred = tf.argmax(prob, axis=1)
                    pred = tf.cast(pred, dtype=tf.int32)

                    # correct: [b], True, False
                    correct = tf.equal(pred, y)
                    correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
                    total_correct += int(correct)
                    total_num += x.shape[0]

                    acc_meter.update_state(y, pred)

                acc = total_correct / total_num
                print(epoch, "Evaluate acc:", acc, acc_meter.result().numpy())


if __name__ == '__main__':
    main()
