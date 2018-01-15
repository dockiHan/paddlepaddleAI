import numpy as np
import pandas as pd
import json
import os

def write_result_to_json(d, file_name, path):
    # the file header in json file
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    this_vid_proposals = []
    for i in d :
        proposal = {
                    'score': d[i],
                    'segment': i,
                   }
        this_vid_proposals += [proposal]

    proposal_data['results'][file_name] = this_vid_proposals
    with open(path, 'a') as fobj:
        json.dump(proposal_data, fobj);

def result_process(df, file_name):
    # delete file if exists because of append mode
    if os.path.exists(file_name):
        os.remove(file_name)
    for col in df.columns:
        single_result = df[col]
        avg = 0.0
        total = 0.0
        start = 0
        end = 0
        count = 0
        d = dict()
        for probability in single_result:
            count += 1 
            total += probability
            avg = total / count 
            end = count - 1
            if (avg < 0.5):
                if (start != end):
                    # get the last interval whose avg score is bigger than 0.5
                    d[(start, end - 1)] = lastavg 
                # clean avg and record new start
                avg = 0.0
                start = count
            lastavg = avg
        if (avg > 0.5):
            d[(start, end)] = avg
        write_result_to_json(d, col, file_name)

# only for test
# if __name__=="__main__":
#     df = pd.DataFrame(np.random.rand(5, 5), columns=['a', 'b', 'c', 'd', 'e'])
#     file_name = "/home/furson/test/test.json"
#     result_process(df, file_name)
