import torch
from torch import nn

input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
input = torch.randn(1, 3, 1000, 1000)
import cv2
image = cv2.imread("../test.jpg")
input = torch.tensor(image.transpose(2, 0, 1))[None,...].float()
print(input.shape)
print(input)


m = nn.ReflectionPad2d(2)
print(m(input))
print(m(input).shape)

m = nn.ReflectionPad2d((1, 0, 1, 0))
print(m(input))
print(m(input).shape)



