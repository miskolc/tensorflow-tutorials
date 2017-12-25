import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b

y = tf.placeholder(tf.float32)


# loss function - MSE
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# initialize graph variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training procedure
training_params = {
    x: x_train,
    y: y_train
}
for i in range(1000):
    sess.run(train, training_params)

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], training_params)
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
