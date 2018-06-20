# coding: utf-8
from Text_CNN import TextCNN
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import os

def predict(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
            sequence_length, vocab_size, embed_size, X_test, y_test, train_epochs,
            initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
            clip_gradients=5.0, decay_rate_big=0.50, dropout=1.0, char_embed_matrix=None):
    with tf.Session() as sess:
        # Instantiate model
        text_CNN = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                           sequence_length, vocab_size, embed_size,
                           initializer=tf.random_normal_initializer(stddev=0.1),
                           multi_label_flag=False, clip_gradients=5.0, decay_rate_big=0.50)
        saver_path = os.getcwd() + '\checkpoint'
        saver = tf.train.Saver(max_to_keep=5)
        model_file = tf.train.latest_checkpoint(saver_path)
        saver.restore(sess, model_file)

        print('Start Testing...')
        feed_dict = {text_CNN.input_x: X_test, text_CNN.input_y: y_test, text_CNN.dropout_keep_prob: dropout,
                     text_CNN.char_embed_matrix: char_embed_matrix, text_CNN.train_iteration: train_epochs,
                     text_CNN.is_train: None}
        acc_test, logits_test = sess.run([text_CNN.accuracy, text_CNN.logits], feed_dict=feed_dict)

        y_pred = np.argmax(logits_test, 1)
        f1_test = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_test))
        # auc_test = roc_auc_score(y_true, y_pred, average='weighted')
        print('The test accuracy / f1 : {0[0]:.2%} / {0[1]:.4f}'.format((acc_test, f1_test)))

if __name__ == '__main__':
    fold_path = os.getcwd() + '\\related_data'

    # Load test data
    lst = ['\X_test.npy', '\y_test.npy']
    X_test, y_test = (np.load(fold_path + name) for name in lst)
    print(len(set(y_test)))

    # Load pre-trained word_embedding
    char_embed_path = fold_path + '\char_embed_matrix.npy'
    if os.path.exists(char_embed_path):
        char_embed_matrix = np.load(char_embed_path)
    else:
        wv_path = fold_path + '\wiki_100_utf8.txt'
        vocab, embed = utils.load_pretrained_wordvector(wv_path)
        char_embed_matrix = np.asarray(embed, dtype='float32')
        np.save(char_embed_path, char_embed_matrix)

    predict(filter_sizes=[3, 4, 5], num_filters=[200, 200, 200], num_classes=78, learning_rate=0.001, batch_size=64,
            decay_steps=0, decay_rate=0, sequence_length=120, vocab_size=16116, embed_size=100, X_test=X_test,
            y_test=y_test, train_epochs=1, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
            clip_gradients=5.0, decay_rate_big=0.50, dropout=1.0, char_embed_matrix=char_embed_matrix)