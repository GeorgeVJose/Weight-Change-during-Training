import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../Dataset/mnist/", one_hot = True)

with tf.name_scope("input"):
    x = tf.placeholder(shape = [None, 784], dtype = tf.float32, name = "x_input")
    y = tf.placeholder(shape = [None, 10], dtype = tf.float32, name = "y_input")


layer1_nodes = 480
layer2_nodes = 680
layer3_nodes = 840
layer4_nodes = 1024
layer5_nodes = 1480
layer6_nodes = 1600
layer7_nodes = 840
layer8_nodes = 480
num_classes = 10


batch_size = 128
num_epochs = 2
learning_rate = 0.0045

summary_path = "./summary_hpc"

def get_weights(shape):
    return tf.Variable(tf.random_normal(shape))

def get_biases(shape):
    return tf.Variable(tf.random_normal(shape))

with tf.name_scope("Layer_1"):
    W_layer1 = get_weights([784, layer1_nodes])
    b_layer1 = get_biases([layer1_nodes])

with tf.name_scope("Layer_2"):
    W_layer2 = get_weights([layer1_nodes, layer2_nodes])
    b_layer2 = get_biases([layer2_nodes])

with tf.name_scope("Layer_3"):
    W_layer3 = get_weights([layer2_nodes, layer3_nodes])
    b_layer3 = get_biases([layer3_nodes])

with tf.name_scope("Layer_4"):
    W_layer4 = get_weights([layer3_nodes, layer4_nodes])
    b_layer4 = get_biases([layer4_nodes])

with tf.name_scope("Layer_5"):
    W_layer5 = get_weights([layer4_nodes, layer5_nodes])
    b_layer5 = get_biases([layer5_nodes])

with tf.name_scope("Layer_6"):
    W_layer6 = get_weights([layer5_nodes, layer6_nodes])
    b_layer6 = get_biases([layer6_nodes])

with tf.name_scope("Layer_7"):
    W_layer7 = get_weights([layer6_nodes, layer7_nodes])
    b_layer7 = get_biases([layer7_nodes])

with tf.name_scope("Layer_8"):
    W_layer8 = get_weights([layer7_nodes, layer8_nodes])
    b_layer8 = get_biases([layer8_nodes])


with tf.name_scope("Output_layer"):
    W_output_layer = get_weights([layer8_nodes, num_classes])
    b_output_layer = get_biases([num_classes])


layer1 = tf.matmul(x, W_layer1) + b_layer1
layer2 = tf.matmul(layer1, W_layer2) + b_layer2
layer3 = tf.matmul(layer2, W_layer3) + b_layer3
layer4 = tf.matmul(layer3, W_layer4) + b_layer4
layer5 = tf.matmul(layer4, W_layer5) + b_layer5
layer6 = tf.matmul(layer5, W_layer6) + b_layer6
layer7 = tf.matmul(layer6, W_layer7) + b_layer7
layer8 = tf.matmul(layer7, W_layer8) + b_layer8
output = tf.matmul(layer8, W_output_layer) + b_output_layer

# Mean Functions
w1_mean = tf.reduce_mean(W_layer1)
tf.summary.scalar("W1_mean" ,w1_mean)

w2_mean = tf.reduce_mean(W_layer2)
tf.summary.scalar("W2_mean" ,w2_mean)

w3_mean = tf.reduce_mean(W_layer3)
tf.summary.scalar("W3_mean" ,w3_mean)

w4_mean = tf.reduce_mean(W_layer4)
tf.summary.scalar("W4_mean" ,w4_mean)

w5_mean = tf.reduce_mean(W_layer5)
tf.summary.scalar("W5_mean" ,w5_mean)

w6_mean = tf.reduce_mean(W_layer6)
tf.summary.scalar("W6_mean" ,w6_mean)

w7_mean = tf.reduce_mean(W_layer7)
tf.summary.scalar("W7_mean" ,w7_mean)

w8_mean = tf.reduce_mean(W_layer8)
tf.summary.scalar("W8_mean" ,w8_mean)

output_mean = tf.reduce_mean(W_output_layer)
tf.summary.scalar("Output_mean" ,output_mean)


with tf.name_scope("Cost_Slave"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))
    # tf.summary.scalar("Cost", cost)

optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cost)
training_step = optimizer.apply_gradients(grads_and_vars)

with tf.name_scope("Accuracy_Slave"):
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy_func = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
    # tf.summary.scalar("Accuracy", accuracy)

# numpy files
w1_mean_npy = []
w2_mean_npy = []
w3_mean_npy = []
w4_mean_npy = []
w5_mean_npy = []
w6_mean_npy = []
w7_mean_npy = []
w8_mean_npy = []
output_mean_npy = []
accuracy_npy = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for j in range(100):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([training_step, cost], feed_dict = {x : epoch_x, y : epoch_y})

            # Run means
            w1_mean_npy.append( [sess.run(w1_mean), (epoch*100+j)] )
            w2_mean_npy.append( [sess.run(w2_mean), (epoch*100+j)] )
            w3_mean_npy.append( [sess.run(w3_mean), (epoch*100+j)] )
            w4_mean_npy.append( [sess.run(w4_mean), (epoch*100+j)] )
            w5_mean_npy.append( [sess.run(w5_mean), (epoch*100+j)] )
            w6_mean_npy.append( [sess.run(w6_mean), (epoch*100+j)] )
            w7_mean_npy.append( [sess.run(w7_mean), (epoch*100+j)] )
            w8_mean_npy.append( [sess.run(w8_mean), (epoch*100+j)] )
            output_mean_npy.append( [sess.run(output_mean), (epoch*100+j)] )

            summary = sess.run(merged)
            summary_writer.add_summary(summary, (epoch*100)+j)
            epoch_loss += c
        print("Epoch : ", epoch+1, " / ", num_epochs, ", Loss : ", epoch_loss)

        accuracy = accuracy_func.eval({x:mnist.test.images, y:mnist.test.labels})
        print("Accuracy : ", accuracy)
        accuracy_npy.append((accuracy, (epoch+1)*100))


np.save("./summary_hpc/w1_mean.npy", np.array(w1_mean_npy))
np.save("./summary_hpc/w2_mean.npy", np.array(w2_mean_npy))
np.save("./summary_hpc/w3_mean.npy", np.array(w3_mean_npy))
np.save("./summary_hpc/w4_mean.npy", np.array(w4_mean_npy))
np.save("./summary_hpc/w5_mean.npy", np.array(w5_mean_npy))
np.save("./summary_hpc/w6_mean.npy", np.array(w6_mean_npy))
np.save("./summary_hpc/w7_mean.npy", np.array(w7_mean_npy))
np.save("./summary_hpc/w8_mean.npy", np.array(w8_mean_npy))
np.save("./summary_hpc/output_mean.npy", np.array(output_mean_npy))
np.save("./summary_hpc/accuracy.npy", np.array(accuracy_npy))
