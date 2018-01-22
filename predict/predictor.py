#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import paddle.v2 as paddle
import pandas as pd
import json
from result_process.result.process import result_process
from model.lstm_classifier import LstmClassify 
 

def read_from_file(filepath):
    img_feature = np.load(filepath)
    normalize_fea = (img_feature - np.mean(img_feature, axis=0)) / np.std(img_feature, axis=0)
    return normalize_fea


def get_file_list(path):
    return np.array(os.listdir(path), dtype=str)


def prepare_predict_data(normalize_fea, 
                       predict_set_num, 
                       time_step, 
                       batch_size, 
                       input_size):
    normalize_fea = np.reshape(normalize_fea, (-1, input_size))
    x_shape = (predict_set_num, time_step, input_size)
    train_x = np.zeros(x_shape, dtype=float)
    assert predict_set_num+time_step <= normalize_fea.shape[0], "predict_set_num is too large."
    for i in range(predict_set_num):
        train_x[i, 0:, 0:] = normalize_fea[i: i+time_step, 0:]
    return train_x

def predictor():
    # Define LSTM params
    batch_size = 200
    time_step = 15
    hide_size = 256
    input_size = 2048
    output_size = 2
    lstm_depth = 5
    mix_hide_lr = 9e-3
    lstm_lr = 9e-3
    drop_out = 0.5 
    model_path = "/home/kesci/work/lstm_model.tar"
    predict_batch = 1000
    
    # Initialize paddle trainer
    paddle.init(use_gpu=False, trainer_count=1, log_error_clipping=True)
    
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
    
    base_path = "/mnt/BROAD-datasets/video/validation"
    file_list = get_file_list(base_path)
    # Define fileID_list and result dict
    fileID_list = []
    res_dict = {}
    # Read data from file
    f_number = 1
    for filename in file_list:
        print("Predicting file number: {}. file: {}".format(f_number, filename))
        f_number += 1
        fileID = filename.split(".")[0]
        fileID_list.append(fileID)
        filepath = os.path.join(base_path, filename)
        normalize_fea = read_from_file(filepath)
        predict_set_num = normalize_fea.shape[0] - time_step
    
        # Prepare predict data
        predict_x = prepare_predict_data(normalize_fea=normalize_fea,  
                                         predict_set_num=predict_set_num, 
                                         time_step=time_step, 
                                         batch_size=batch_size, 
                                         input_size=input_size)
        predict_x = np.reshape(predict_x, (predict_set_num, 1, time_step, input_size))
        # Generating predict data by 'predict_batch' size and predicting
        if predict_set_num % predict_batch == 0:
            num_of_batch = predict_set_num // predict_batch
        else:
            num_of_batch = predict_set_num // predict_batch + 1
        tmp_feature = predict_x[:predict_batch]
        predict_result = lstm.lstm_predict(features=tmp_feature, load_path=model_path)
        for i in range(1, num_of_batch):
            tmp_feature = predict_x[i * predict_batch: (i + 1) * predict_batch]
            tmp_result = lstm.lstm_predict(features=tmp_feature, load_path=model_path)
            predict_result = np.concatenate((predict_result, tmp_result), axis=0)
        
        # Padding the first 'time_step' labels
        padding = np.zeros(time_step)
        predict_result = np.concatenate((padding, predict_result), axis=0)
        # Test: Count the predict_result(prob > 0.5)
        """
        count = 0
        for prob in predict_result:
            if prob > 0.5:
                count += 1
        print("Count: {}".format(count))
        print(predict_result)
        """
        tmp_dict = { 
                     fileID: predict_result 
                   }
        
        res_dict.update(tmp_dict)
    # Process predict result    
    result_path = "/home/kesci/work/validation.json"
    result_process(res_dict, result_path)
    # Write result to file
    """
    predict_path = "/home/kesci/work/predict_result.json"
    with open(predict_path, 'w') as file:
        json.dump(res_dict, file)
    """

if __name__ == "__main__":
    predictor()
    
