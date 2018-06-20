# coding: utf-8
import utils
from Text_CNN import TextCNN
import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

batch_index = 0
epoch = 1
def next_batch(X_train, y_train, batch_size):
    global epoch
    global batch_index

    start = batch_index
    n_example = X_train.shape[0]
    batch_index += batch_size

    if batch_index >= n_example:
        epoch += 1 # run the new epoch
        batch_index = 0
        start = batch_index
        batch_index += batch_size
        rand = [i for i in range(n_example)]
        np.random.shuffle(rand)
        X_train = X_train[rand]
        y_train = y_train[rand]

    assert batch_size < n_example
    end = batch_index

    return X_train[start: end], y_train[start: end]

# def visualization(step_range, train_range, dev_range, y_max, y_min, y_label):
#     plt.plot(step_range, train_range, label='train')
#     plt.plot(step_range, dev_range, label='dev')
#     plt.legend(loc='lower right', frameon=False)
#     plt.ylim(ymax=y_max, ymin=y_min)
#     plt.ylabel(y_label)
#     plt.xlabel('Steps')
#     plt.show()

def train(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
          sequence_length, vocab_size, embed_size, X_train, y_train, is_dev, X_dev, y_dev, train_epochs,
          initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False, clip_gradients=5.0,
          decay_rate_big=0.50, dropout=0.5, char_embed_matrix=None):
    with tf.Session() as sess:
        # Instantiate model
        text_CNN = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                        sequence_length, vocab_size, embed_size, initializer=tf.random_normal_initializer(stddev=0.1),
                        multi_label_flag=False, clip_gradients=5.0, decay_rate_big=0.50)
        # Initialize Save
        saver = tf.train.Saver(max_to_keep=5)
        saver_path = os.getcwd() + '\checkpoint'
        # if os.path.exists(saver_path + '\model.ckpt'):
        #     print('Restoring Variables from Checkpoint')
        #     saver.restore(sess, tf.train.latest_checkpoint(saver_path))
        # else:
        #     print('Initializing variables')
        #     sess.run(tf.global_variables_initializer())
        print('Start Training......')
        sess.run(tf.global_variables_initializer())

        # Feed data & Training & Visualization
        train_losses, dev_losses = [], []
        train_accuracies, dev_accuracies = [], []
        step_range = []
        display_step = 20
        total_step = 0
        min_loss = 100
        while epoch <= train_epochs:
            X_batch, y_batch = next_batch(X_train, y_train, batch_size)
            step = batch_index // batch_size
            if step % display_step ==0:
                feed_dict = {text_CNN.input_x: X_batch, text_CNN.input_y: y_batch,
                             text_CNN.dropout_keep_prob: dropout,
                             text_CNN.char_embed_matrix: char_embed_matrix, text_CNN.train_iteration: train_epochs,
                             text_CNN.is_train: True}
                train_loss, train_accuracy = sess.run([text_CNN.loss_val, text_CNN.accuracy], feed_dict=feed_dict)
                if (is_dev):
                    feed_dict = {text_CNN.input_x: X_dev, text_CNN.input_y: y_dev,
                                 text_CNN.dropout_keep_prob: 1.0, text_CNN.char_embed_matrix: char_embed_matrix,
                                 text_CNN.train_iteration: train_epochs, text_CNN.is_train: None}
                    dev_loss, dev_accuracy = sess.run([text_CNN.loss_val, text_CNN.accuracy], feed_dict=feed_dict)
                    print('Epoch %d: train_loss / dev_loss => %.4f / %.4f for step %d' % (epoch, train_loss, dev_loss, step))
                    print('Epoch {0[0]}: train_accuracy / dev_accuracy => {0[1]:.2%} / {0[2]:.2%} for step {0[3]}'.format(
                        (epoch, train_accuracy, dev_accuracy, step)))

                    if dev_loss < min_loss:
                        saver.save(sess, saver_path + '\\vali_loss_{:.4f}.ckpt'.format(dev_loss))
                        min_loss = dev_loss

                    dev_losses.append(dev_loss)
                    dev_accuracies.append(dev_accuracy)

                else:
                    print('Epoch %d: train_loss => %.4f for step %d' % (epoch, train_loss, step))
                    print('Epoch {0[0]}: train_accuracy => {0[1]:.2%} for step {0[2]}'.format(epoch, train_accuracy, step))
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                total_step += display_step
                step_range.append(total_step)

            # train on batch
            feed_dict = {text_CNN.input_x: X_batch, text_CNN.input_y: y_batch, text_CNN.dropout_keep_prob: dropout,
                         text_CNN.char_embed_matrix: char_embed_matrix, text_CNN.train_iteration: train_epochs,
                         text_CNN.is_train: True}
            sess.run(text_CNN.train_op, feed_dict=feed_dict)

        # visualization(step_range, train_losses, dev_losses, 2, -1, 'Loss')
        # visualization(step_range, train_accuracies, dev_accuracies, 1.1, 0.5, 'Accuracy')
    sess.close()


if __name__ == '__main__':
    fold_path = os.getcwd() + '\\related_data'

    # Load pre-trained word_embedding
    char_embed_path = fold_path + '\char_embed_matrix.npy'
    if os.path.exists(char_embed_path):
        char_embed_matrix = np.load(char_embed_path)
    else:
        wv_path = fold_path + '\wiki_100_utf8.txt'
        vocab, embed = utils.load_pretrained_wordvector(wv_path)
        char_embed_matrix = np.asarray(embed, dtype='float32')
        np.save(char_embed_path, char_embed_matrix)

    # Load train&dev data
    lst = ['\X_train.npy', '\y_train.npy', '\X_dev.npy', '\y_dev.npy']
    X_train, y_train, X_dev, y_dev = (np.load(fold_path + name) for name in lst)

    train(filter_sizes=[3, 4, 5], num_filters=[200, 200, 200], num_classes=78, learning_rate=0.001, batch_size=64,
          decay_steps=0, decay_rate=0, sequence_length=120, vocab_size=16116, embed_size=100, X_train=X_train,
          y_train=y_train, is_dev=True, X_dev=X_dev, y_dev=y_dev, train_epochs=50,
          initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False, clip_gradients=5.0,
          decay_rate_big=0.50, dropout=0.5, char_embed_matrix=char_embed_matrix)
