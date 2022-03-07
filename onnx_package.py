from facemesh import FaceMesh
import torch

net = FaceMesh()
net.load_weights("facemesh.pth")
torch.onnx.export(
    net,
    torch.randn(1, 3, 192, 192, device='cpu'),
    "facemesh.onnx",
    input_names=("image", ),
    output_names=("preds", "confs"),
    opset_version=9
)