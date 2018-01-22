import numpy as np
import pandas as pd
import json
import os

def write_result_to_json(d, path):
    # the file header in json file
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    for v_id in d :
        this_vid_proposals = []
        v_dist = d[v_id]
        for v_list in v_dist:
            proposal = {
                        'score': v_list[1],
                        'segment': v_list[0],
                       }
            this_vid_proposals += [proposal]

        proposal_data['results'][v_id] = this_vid_proposals
    with open(path, 'w') as fobj:
        json.dump(proposal_data, fobj);

def result_process(in_d, file_name):
    # delete file if exists because of append mode
    if os.path.exists(file_name):
        os.remove(file_name)
    res_dict = dict()
    for v_id in in_d:
        v_res = in_d[v_id]
        index = 0          # index in v_res
        start = index      # the start index of a wonderful interval
        end = index        # the last index plus one of a wonderful interval
        seg_sum = 0.0      # the interval's sum
        last_prob = 0.0    # only record the last prob
        one_video_dict = dict()
        for prob in v_res:
            index += 1
            if (prob < 0.5):
                # interval is more than one frame
                if (start != end and start + 1 != end):
                    # the wonderful interval should be [start, end - 1]
                    one_video_dict[(start, end - 1)] = seg_sum / (end - start) 
                # clean seg_sum and record new start
                seg_sum = 0.0
                start = index
                end = index
            else:
                seg_sum += prob
                end = index 
            last_prob = prob
        # tail frames are all wonderful frames
        if (last_prob > 0.5):
            if (start != end and start + 1 != end):
                one_video_dict[(start, end - 1)] = seg_sum / (end - start)
        one_video_dict = sorted(one_video_dict.items(), lambda d1, d2: cmp(d1[0][0], d2[0][0]))
        res_dict[v_id] = one_video_dict
    write_result_to_json(res_dict, file_name)
    

# only for test
"""
if __name__=="__main__":
    file_path = "/home/kesci/work/predict_result.json"
    with open(file_path, "r") as f:
        json_str = json.load(f)
    file_name = "/home/kesci/work/validation.json"
    result_process(json_str, file_name)
    aarr = np.random.random(size=50)
    barr = np.random.random(size=4)
    carr = np.random.random(size=9)
    darr = np.random.random(size=7)
    d = {'a': aarr, 'b': barr,
          'c': carr, 'd': darr}
    print(d)
    file_name = "/home/furson/test/test.json"
    result_process(d, file_name)
"""
