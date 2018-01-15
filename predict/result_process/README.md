# Result Processing

## How to Use
Import it and add use result\_process function:
First arg is pandas.dataframe
Second arg is file\_name and you should add ".json" in this name

## Attention
1. If the interval's avg score is bigger than 0.5, considering it's a wonderful interval and append it to the json file. (Not a good evaluation way)
2. This way to process pandas.dataframe maybe cause low speed, if you have better process methods, please modify it.
3. The second arg should add suffix '.json'.
