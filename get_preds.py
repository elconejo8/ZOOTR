import cv2
from functions import shave_black_bit, get_frames_from_video
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from pytorch_efficientnet import to_tensor, EfficientNetwork, eval_model, augment, augmented_pred
import torch
from aux_funs import save_video_frames

model = EfficientNetwork(output_size=2, b1=False, b2=True)
model.load_state_dict(torch.load('Models\\model_17.pth'))
model.eval()

predictions = []
img_batch = []
vid = 'C:\\Users\\Guilherme\\Videos\\Captures\\Nintendo 64 - OoTR 593579 UF41FK8V98 2020-10-19 12-53-01.mp4'
vid = 'C:\\Users\\Guilherme\\Videos\\Captures\\424268\\trimmed_vid.mp4'
vid = 'C:\\Users\\Guilherme\\Videos\\Captures\\424268\\New video.mp4'

predictions = save_video_frames(vid, 'Testes', 1, model, to_tensor)
predictions = save_video_frames(vid, 'Testes', 3, model, augmented_pred)