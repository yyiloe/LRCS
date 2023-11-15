import numpy as np 
from matplotlib import pyplot as plt 
import os
root_path=os.path.dirname(__file__)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}

x = [1, 2, 3, 4]
y = [14.83, 16.67, 18.57, 19.34]
plt.plot(x,y,marker = "o",markersize=6)
x = [1, 2, 3, 4]
y = [11.25, 11.85, 12.12, 12.95]
plt.plot(x,y,marker = "o",markersize=6)
x = [1, 2, 3, 4]
y = [12.05, 12.34, 12.72,13.21]

plt.xlabel("Embedding Size",fontdict=font1) 
plt.ylabel("Scores",fontdict=font1) 
plt.plot(x,y,marker = "o",markersize=6)

# x = [1, 2, 3, 4]
# y = [15.75, 17.21, 18.76, 19.34]
# plt.plot(x,y,marker = "o",markersize=6)
# x = [1, 2, 3, 4]
# y = [10.75, 11.23, 12.01, 12.95]
# plt.plot(x,y,marker = "o",markersize=6)
# x = [1, 2, 3, 4]
# y = [11.94, 12.21, 12.87, 13.21]

# plt.xlabel("Num Layers",fontdict=font1) 
# plt.ylabel("Scores",fontdict=font1) 
# plt.plot(x,y,marker = "o",markersize=6) 

plt.legend(['C-BLEU','S-BLEU','METEOR'],prop=font1)
plt.xticks(x,[1,2,3,4])
plt.ylim(10, 20)
plt.savefig(os.path.join(root_path,'embs.png'))