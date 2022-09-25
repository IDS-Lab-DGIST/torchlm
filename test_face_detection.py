import torchlm
from torchlm.tools import faceboxesv2, yolov5face
from torchlm.models import pipnet
import cv2


image = cv2.imread("test.jpg")

# torchlm.runtime.bind(faceboxesv2(device="cpu"))  # set device="cuda" if you want to run with CUDA
torchlm.runtime.bind(yolov5face())  # set device="cuda" if you want to run with CUDA
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(
  pipnet(backbone="resnet18", pretrained=True,  
         num_nb=10, num_lms=98, net_stride=32, input_size=256,
         meanface_type="wflw", map_location="cpu", checkpoint=None) 
) # will auto download pretrained weights from latest release if pretrained=True
landmarks, bboxes = torchlm.runtime.forward(image)
image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
# image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)

# save image to disk
cv2.imwrite("test_out.jpg", image)
