import tensorflow as tf
import numpy as np

iris_file_path = './data.csv'
logdir = './logs/4/'

BATCH_SIZE = 5
num_epochs = 500

# function to retreive data and label tensors
# setup input pipeline for iris
def get_data():
    # read csv values into np array
    data = np.genfromtxt(iris_file_path,delimiter=',')
    
    # read features into a separate np array
    features_list = [list(rec)[:4] for rec in data]
    features = np.array(features_list) 
    
    # read labels into a separate np array
    labels_list = [list(rec)[4] for rec in data]
    labels = np.array(labels_list).astype(np.int32)

    # convert labels to one-hot vectors for easy processing
    labels = _convert_to_one_hot(labels)
    
    return features,labels

def _convert_to_one_hot(vector):
    one_hot_vector = np.zeros([vector.shape[0],3])
    for idx in range(len(vector)):
        one_hot_vector[idx][vector[idx]-1]=1
    return one_hot_vector

def _get_next_batch(features,labels,batch_idx):
    batch_features = features[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
    batch_labels = labels[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
    
    return batch_features,batch_labels

def _add_layer(features,input_channels,output_channels,scope):
    with tf.variable_scope(scope):
        # weight initializer
        initw = tf.truncated_normal([input_channels,output_channels])
        weight = tf.get_variable('weight',initializer=initw)

        # bias initializer
        initb = tf.constant(0.0,shape=[output_channels])
        bias = tf.get_variable('bias',initializer=initb)

        out = tf.matmul(features,weight) + bias

        tf.summary.histogram('weight',weight)
        tf.summary.histogram('bias',bias)
        tf.summary.histogram('out',out)

    return out

def neural_network(features):
    # add a simple feed forward layer

    input_channels = int(features.get_shape()[-1])
    out1 = _add_layer(features,input_channels,10,'layer1')

    input_channels = int(out1.get_shape()[-1])
    output = _add_layer(out1,input_channels,3,'layer2')
    
    output = tf.nn.softmax(output)
    return output
    
def train():
    
    sess = tf.Session()

    # declare placeholder for features and labels
    X = tf.placeholder(tf.float32,[None,4], name='features')
    Y = tf.placeholder(tf.float32,[None,3], name='labels')

    # get the data and labels
    features ,labels = get_data()
    
    with tf.variable_scope('network'):
        # get the output from the neural network
        prediction= neural_network(X)

    # define the loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))
        tf.summary.scalar('loss',loss)
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer().minimize(loss)
    
    num_batches = len(features)/BATCH_SIZE

    sess.run(tf.global_variables_initializer())

    # write our graph for tensorboard
    writer= tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)
    

    merged = tf.summary.merge_all()

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # get the next batch
            batchX,batchY = _get_next_batch(features,labels,batch)

            # write all summaries to tensorboard log-dir
            if (batch%5==0):
                s = sess.run(merged, feed_dict={X:batchX,Y:batchY})
                writer.add_summary(s)

            # run the graph for training 
            _,epoch_loss= sess.run([train_step,loss],feed_dict={X:batchX,Y:batchY})


            print "Epoch " + str(epoch) + "/" + str(num_epochs) + "; Batch " + str(batch) + "/" + str(num_batches) + " completed ; loss : " + str(epoch_loss)

    with tf.variable_scope('network') as scope:
        scope.reuse_variables()
        test(sess,features,labels)

    sess.close()


def test(sess,features,labels):
    test_features  = features[:5]  
    test_labels = labels[:5]

    X = tf.placeholder(tf.float32,[None,4])
    Y = tf.placeholder(tf.float32,[None,3])

    # find the predictions 
    prediction = neural_network(X)
    
    pred = sess.run(prediction , feed_dict={X:test_features})

    print pred

train()

