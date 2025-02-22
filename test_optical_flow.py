import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image



cap = cv2.VideoCapture('/home/techtrans2/RAT_DATASETS/LAB_RAT_ACTIONS_DATASET/grooming/videos/grooming_4.mp4')
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame)
# hsv[..., 1] = 255
# while(1):
#     ret, frame = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break
#     next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     cv2.imshow('frame', bgr)
#     if cv2.waitKey(0) == ord('q'):
#         break
#     prvs = next

# cv2.destroyAllWindows()


transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_small(pretrained=True, progress=False).to(device)
model.eval()

old_frame = torch.from_numpy(frame).unsqueeze(0).to(device).permute(0, 3, 1, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cv2.waitKey(0) == ord('q'):
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).unsqueeze(0).to(device).permute(0, 3, 1, 2)

    pred_flow = model(transforms(old_frame), transforms(frame))[-1]
    pred_flow = flow_to_image(pred_flow).cpu().squeeze().numpy().transpose(1, 2, 0)
    cv2.imshow('rat', pred_flow.astype(np.uint8))
    old_frame = frame