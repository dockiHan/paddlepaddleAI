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
        for i in v_dist:
            proposal = {
                        'score': v_dist[i],
                        'segment': i,
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
                if (start != end):
                    # the wonderful interval should be [start, end - 1]
                    one_video_dict[(start, end - 1)] = seg_sum / (end - start) 
                # interval is only one frame
                elif (seg_sum != 0.0):
                    one_video_dict[(start, end)] = seg_sum
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
            if (start != end):
                one_video_dict[(start, end - 1)] = seg_sum / (end - start)
            else:
                one_video_dict[(start, end)] = seg_sum
        res_dict[v_id] = one_video_dict
    # print(res_dict)
    write_result_to_json(res_dict, file_name)
    

# only for test
# if __name__=="__main__":
#     aarr = np.random.random(size=1)
#     barr = np.random.random(size=4)
#     carr = np.random.random(size=2)
#     darr = np.random.random(size=7)
#     d = {'a': aarr, 'b': barr,
#           'c': carr, 'd': darr}
#     print(d)
#     file_name = "/home/furson/test/test.json"
#     result_process(d, file_name)
