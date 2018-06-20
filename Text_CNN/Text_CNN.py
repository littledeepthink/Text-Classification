# coding: utf-8
import numpy as np
import tensorflow as tf

class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, initializer=tf.random_normal_initializer(stddev=0.1),
                 multi_label_flag=False, clip_gradients=5.0, decay_rate_big=0.50):
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        # self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes # a list of int. e.g. [3,4,5]
        self.num_filters = num_filters # a list of int. e.g. [200, 200, 400]
        self.initializer = initializer
        self.num_filters_total = sum(self.num_filters) #how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        # self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.char_embed_matrix = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size], name="char_embed_matrix")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.train_iteration = tf.placeholder(tf.int32) # training iteration
        self.is_train = tf.placeholder(tf.bool) # train or test

        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step=tf.Variable(0, trainable=False, name="Epoch_Step")
        # self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step,tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters[0]]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters[0]]) / 10)

        # self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.variables_define()
        self.logits = self.model() #[None, self.label_size]. main computation graph is here.
        self.possibility = tf.nn.softmax(self.logits)

        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

        self.train_op = self.train()

        if not self.multi_label_flag:
            self.label_pred = tf.argmax(self.logits, 1, name='label_pred') # shape:[None,]
            # self.label_true = tf.argmax(self.input_y, 1, name='label_true')
            # print("self.predictions:", self.label_pred)
            correct_prediction = tf.equal(tf.cast(self.label_pred, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def variables_define(self):
        """define all weights here"""
        with tf.name_scope("variable"):
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            if self.is_train is True:
                self.Embedding.assign(self.char_embed_matrix)
            self.W = tf.get_variable("W", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer) #[embed_size,label_size]
            self.b = tf.get_variable("b", shape=[self.num_classes])

    def model(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)#[None,sentence_length,embed_size]
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.embedded_sentences = tf.expand_dims(self.embedded_words, -1)

        # 2.loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # a.Create filter
                filter = tf.get_variable("filter-%s"%filter_size, [filter_size, self.embed_size, 1, self.num_filters[i]],
                                         initializer=self.initializer)
                # b.Conv operation: conv2d
                # shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]
                conv = tf.nn.conv2d(self.embedded_sentences, filter, strides=[1,1,1,1], padding="VALID", name="conv")
                # c.Batch Normalization
                self.is_test = tf.constant(self.is_train is None, dtype=tf.bool)
                conv, self.update_ema = self.batch_norm(conv, self.is_test, self.train_iteration, self.b1)
                # d.Acticvate function
                bias = tf.get_variable("bias-%s"%filter_size, [self.num_filters[i]])
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), "relu")
                # e.Max-pooling.
                # shape:[batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # 3.combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.
        # where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().
        # e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        #4.add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob) #[None, num_filters_total]
        self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        #5.logits(use linear layer) and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W) + self.b # shape:[None, self.num_classes]
        return logits

    def batch_norm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        # adding the iteration prevents from averaging across non-existing iterations
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss_multilabel(self,l2_lambda=0.0001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses = tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)         #shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
        return loss

    def loss(self, l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size, num_classes]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
        return loss

    def train(self):
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        # train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
        #                                            learning_rate=learning_rate, optimizer="Adam",
        #                                            clip_gradients=self.clip_gradients)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads = optimizer.compute_gradients(self.loss_val)
        clip_grads = [(tf.clip_by_value(grad, -self.clip_gradients, self.clip_gradients), var) for grad, var in grads]
        train_op = optimizer.apply_gradients(clip_grads)
        return train_op