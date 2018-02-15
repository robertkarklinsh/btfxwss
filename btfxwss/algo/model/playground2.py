import tensorflow as tf


def define_graph(input):
    W = tf.Variable(tf.constant([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]], dtype=tf.float32, shape=[3, 3]))
    b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[3]))
    output = W * input + b
    return output, b


if __name__ == '__main__':
    input = tf.constant(1.0, dtype=tf.float32, shape=[3])
    output, bias = define_graph(input)
    grad = tf.gradients(output, [input, bias])# grad_ys=tf.constant(1., dtype=tf.float32, shape=[3, 3]))
    with tf.Session() as sess:




        tf.global_variables_initializer().run()
        output_value, grad_value = sess.run([output, grad])
        print(output_value)
        print(grad_value)
