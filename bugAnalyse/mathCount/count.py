import math
import torch

diff = 0.0461
diff = -0.0018
diff = 0.0010
diffs = [0.0451,-0.0018,0.0010]
for diff in diffs:
    diff *=100
    result = 1 / (1 + math.exp(-diff))
    print(result)
    result2 = - math.log(result + 1e-8)
    print(result2)