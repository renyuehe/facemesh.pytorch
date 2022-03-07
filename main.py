import torch
import cv2
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from facemesh import FaceMesh
net = FaceMesh().to(gpu)
net.load_weights("facemesh.pth")

img = cv2.imread("test2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 192))

detections = net.predict_on_image(img).cpu().numpy()
print(detections.shape)

import matplotlib.pyplot as plt
plt.imshow(img, zorder=1)
x, y = detections[:, 0], detections[:, 1]
plt.scatter(x, y, zorder=2, s=1.0)
plt.show()