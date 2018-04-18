#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz


tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./rnn_log',
                        help='path to save log and checkpoint.')

    parser.add_argument('--text', type=str, default='/data/jieming2002/quansongci-data/QuanSongCi.txt',
                        help='path to QuanSongCi.txt')

    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of time steps of one sample.')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch size to use.')

    parser.add_argument('--dictionary', type=str, default='/data/jieming2002/quansongci-data/dictionary.json',
                        help='path to dictionary.json.')

    parser.add_argument('--reverse_dictionary', type=str, default='/data/jieming2002/quansongci-data/reversed_dictionary.json',
                        help='path to reverse_dictionary.json.')
    
    parser.add_argument('--embedding_file', type=str, default='/data/jieming2002/quansongci-data/embedding.npy',
                        help='path to embedding.npy.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    
    parser.add_argument('--keep_prob', type=float, default=0.9,
                        help='keep_prob')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    for x in dir(FLAGS):
        print(getattr(FLAGS, x))
