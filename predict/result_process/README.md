# Result Processing

## How to Use
Import it and add use result\_process function:
First arg is a dist, the dist should be [file\_name, numpy.array], the key is the video\id, the value is the video's each frame's value.
Second arg is file\_name and you should add ".json" in this name

## Attention
1. This file would record interval when meet one frame's prob less than 0.5
   check whether index start equals with index end:
if __true__ and have already read one prob, record it, just like [4, 4] and its prob(of course, bigger than or equal with 0.5)
if __false__ record [start, end - 1] and their prob.
2. This way to process pandas.dataframe maybe cause low speed, if you have better process methods, please modify it.
3. The second arg should add suffix '.json'.
4. Only run simple test for this, better test it more before using.
