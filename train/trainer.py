#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import paddle.v2 as paddle
import gc
from model.lstm_classifier import LstmClassify
from data_prepare.pre_processing import getLabelArray 

def read_from_file(filepath):
    img_feature = np.load(filepath)
    normalize_fea = (img_feature - np.mean(img_feature, axis=0)) / np.std(img_feature, axis=0)
    return normalize_fea


def get_file_list(path):
    return np.array(os.listdir(path), dtype=str)


def prepare_train_data(normalize_fea, 
                       label, 
                       train_set_num, 
                       time_step, 
                       batch_size, 
                       input_size):
    normalize_fea = np.reshape(normalize_fea, (-1, input_size))
    assert time_step + train_set_num <= label.shape[0], "time_step or train_set_num is too large."
    train_y = label[time_step: time_step+train_set_num].astype(int)
    x_shape = (train_set_num, time_step, input_size)
    train_x = np.zeros(x_shape, dtype=float)
    for i in range(train_set_num):
        train_x[i, 0:, 0:] = normalize_fea[i: i+time_step, 0:]
    return train_x, train_y


def trainer():
    # Define LSTM params
    batch_size = 200
    time_step = 15
    hide_size = 256
    input_size = 2048
    output_size = 2
    lstm_depth = 5
    mix_hide_lr = 9e-3/batch_size
    lstm_lr = 9e-3/batch_size
    drop_out = 0.5 
    train_pass_num = 3000 
    model_path = "/home/docki/work/video/lstm_model.tar"
    
    # Initialize paddle trainer
    paddle.init(use_gpu=True, trainer_count=1, log_error_clipping=True)
    
    # Initialize lstm model
    lstm = LstmClassify(batch_size=batch_size,
                        time_step=time_step,
                        hide_size=hide_size,
                        input_size=input_size,
                        output_size=output_size,
                        lstm_depth=lstm_depth,
                        mix_hide_lr=mix_hide_lr,
                        lstm_lr=lstm_lr,
                        drop_out=drop_out)
    
    # Read data from file
    base_path = "/home/docki/work/video/training"
    file_list = get_file_list(base_path)
    file_index = 0
    file_batch = 8
    while file_index < len(file_list):
        filename = file_list[file_index]
        fileID = filename.split(".")[0]
        filepath = os.path.join(base_path, filename)
        normalize_fea = read_from_file(filepath)
        label = getLabelArray(fileID).astype(int)
        print("File number: {}. file: {}".format(file_index+1, filename))
        iteration = 1
        for filename in file_list[file_index+1: file_index+file_batch]:
            print("File number: {}. file: {}".format(file_index+1+iteration, filename))
            iteration += 1
            fileID = filename.split(".")[0]
            filepath = os.path.join(base_path, filename)
            tmp_normalize_fea = read_from_file(filepath)
            normalize_fea = np.concatenate((normalize_fea, tmp_normalize_fea), axis=0)
            tmp_label = getLabelArray(fileID).astype(int)
            label = np.concatenate((label, tmp_label), axis=0)
        train_set_num = normalize_fea.shape[0] - time_step
        print(normalize_fea.shape)
        print(label.shape)
        # Prepare train data
        train_x, train_y = prepare_train_data(normalize_fea=normalize_fea, 
                                              label=label, 
                                              train_set_num=train_set_num, 
                                              time_step=time_step, 
                                              batch_size=batch_size, 
                                              input_size=input_size)
        # Training
        increment = (False if file_index == 0 else True)
        last_saved_cost = 10000.0
        if file_index > 0:
            last_saved_cost = lstm.get_last_saved_cost()
        init_model = model_path
        lstm.lstm_train(train_set_num, train_x, train_y, model_path, init_model, 
                        increment, 10, train_pass_num, last_saved_cost)
        print("*" * 66)
        print("Last saved cost: {}".format(last_saved_cost))
        print("*" * 66)
        file_index += file_batch
        
if __name__ == "__main__":
    trainer()
