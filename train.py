#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf
import numpy as np

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

# 读取文本，得到的是一个字符列表，每个元素是单个字符
vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

# 字典中，key 是单词，value 是索引 
with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

# 逆向字典中，key 是索引，value 是单词 
with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
# model.build()
model.build(FLAGS.embedding_file)


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')
    # 先把单词转换为索引，下面训练时，就不用每次都转换了 
    vocabulary_ix = utils.index_data(vocabulary, dictionary)

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        
        for dl in utils.get_train_data(vocabulary_ix, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
            ##################
            # Your Code here
            ##################
            # dl 是一个元组（data，label） ，data 和 label 的元素是单词列表
            #logging.debug('dl = {0}'.format(dl))
            # logging.debug('here {0}....'.format(x))
            
            # 把单词转换为字典索引，才能用于训练
            data_batch = [utils.index_data(ch, dictionary) for ch in dl[0]]
            label_batch = [utils.index_data(ch, dictionary) for ch in dl[1]]
            #print('data_batch =', data_batch)
            #print('label_batch =', label_batch)
            feed_dict = {model.X:data_batch, model.Y:label_batch, model.state_tensor:state, model.keep_prob:FLAGS.keep_prob}
            
            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 10 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
