import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap
import functools

import torch
from collections.abc import Callable

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_kinetics import Feeder_kinetics

toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")

def split_data_and_label(data, label, s, window_len):

    data_segment_list = []
    label_segment_list = []
    name_segment_list = []

    time_lengh = data.shape[1]
    segment_num = time_lengh//window_len
    for i in range(segment_num):
        data_segment_list.append(data[:, i*window_len:(i+1)*window_len,:,:])
        label_segment_list.append(label)
        name_segment_list.append('segment_'+str(i)+'_'+s)
    return data_segment_list, label_segment_list, name_segment_list, segment_num

def gendata(
        data_path,
        # label_path,
        data_out_path,
        label_out_path,
        num_person_in=2,  #observe the first 5 persons 
        num_person_out=1,  #then choose 2 persons with the highest score 
        max_frame=25):

    feeder = Feeder_kinetics(data_path=data_path,
                             # label_path=label_path,
                             num_person_in=num_person_in,
                             num_person_out=num_person_out,
                             window_size=max_frame)
    
    total_window_num = feeder.window_num
    sample_name = feeder.sample_name
    
    sample_label = []
    sample_name_segment = []
    # print("len(sample_name)", len(sample_name))
    print("total_window_num", total_window_num)

    fp = open_memmap(data_out_path,
                     dtype='float32',
                     mode='w+',
                     shape=(total_window_num, 3, max_frame, 42,
                            num_person_out))
    window_num_count = 0
    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print("data, label", data.shape, label)
        data_segment_list, label_segment_list, name_segment_list, segment_num = split_data_and_label(data, label, s, max_frame)
        data_segment_array = np.stack(data_segment_list, axis=0)
        
        print_toolbar(
            i * 1.0 / len(sample_name),
            '({:>5}/{:<5}) Processing data: '.format(i + 1, len(sample_name)))
        fp[window_num_count: window_num_count+segment_num, :, 0:data.shape[1], :, :] = data_segment_array
        
        sample_label.extend(label_segment_list)
        sample_name_segment.extend(name_segment_list)
        window_num_count += segment_num
        print("window_num_count", window_num_count)
        
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name_segment, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument('--data_path',
                        default='/home/cqy/Documents/cqy/handwashing/2_hand_hygiene/dataset-preprocessed/videos')
    parser.add_argument('--out_folder',
                        default='data/handwash')
    arg = parser.parse_args()

    part = ['trainval','test']
    for p in part:
        data_path = '{}/{}'.format(arg.data_path, p)
        # label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, data_out_path, label_out_path)
