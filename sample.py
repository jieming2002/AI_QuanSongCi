#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import tensorflow as tf

import utils
from model import Model

from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')


reverse_list = [reverse_dictionary[str(i)]
                for i in range(len(reverse_dictionary))]
titles = ['江神子', '蝶恋花', '渔家傲']


model = Model(learning_rate=FLAGS.learning_rate, batch_size=1, num_steps=1)
model.build(FLAGS.embedding_file)

with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')
        exit(0)

    for title in titles:
        state = sess.run(model.state_tensor)
        # feed title
        for head in title:
            input = utils.index_data(np.array([[head]]), dictionary)

            feed_dict = {model.X: input,
                         model.state_tensor: state,
                         model.keep_prob: 1.0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

        sentence = title
        word_index = pred[0].argsort()[-1]

        # generate sample
        for i in range(64):
            feed_dict = {model.X: [[word_index]],
                         model.state_tensor: state,
                         model.keep_prob: 1.0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

            word_index = pred[0].argsort()[-1]
            word = np.take(reverse_list, word_index)
            sentence = sentence + word

        logging.debug('==============[{0}]=============='.format(title))
        logging.debug(sentence)
