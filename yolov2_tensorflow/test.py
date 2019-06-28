import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
import numpy as np
import torch
from torch.autograd import Variable

data = np.arange(1, 5).reshape((2, 2))
z = Variable(torch.randn(5, 5), requires_grad=True)
print(z)
data1 = torch.from_numpy(data)
data2 = data1.numpy()

print(data)
print(data1)
print(data2)