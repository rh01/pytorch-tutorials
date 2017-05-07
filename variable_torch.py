import torch
import numpy as np
from torch.autograd import Variable
data = [[1, 2],[3, 4]]

data = torch.FloatTensor(data)
variable = Variable(data, requires_grad=True)
# variable = variable.
print(data)
print(variable)



t_out = torch.mean(data*data)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)


v_out.backward()
# v_out= 1/4 * sum(var*var)
# d(v_out) / d(var) = 1/4 * 2 * var = (var)/2
print(variable)
print(variable.grad)

# cann't do this
# print(variable.numpy())
# should be in this way
print(variable.data.numpy()) # transform tensor to numpy data structure

