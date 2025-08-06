import cv2
import os
import torch
from torch import softmax
from torch.autograd import Variable
from PIL import Image

def get_image_pred(img, model, transform = None):
#    x = Variable(torch.from_numpy(img))
#    x.to('cpu', dtype=torch.float)
    x = img
    if transform is not None:
        x = transform(x)
    out = model(x[None, :, :, :])
    pred = softmax(out, 1).cpu().detach().numpy()[0]
    return pred[1]
#Get prediction on a single image
    


def save_video_frames(vid, save_to_path, n,  model=None, transform=None):
    video_capture = cv2.VideoCapture(vid)
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions =  []
    for j in range(v_len):
        success, vframe = video_capture.read()
        if j % n == 0:
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            if model is not None:
                pred = get_image_pred(vframe, model, transform)
                predictions.append(float(pred))
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '-' + str(float(pred)) + '.png'))
            else:
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '.png'))
    return predictions
#Gets a video and saves every n frame to specified folder. Frame name is it's sequential number. 
#If given a model it calculates probability of frame being an item and appends it to the name.
#Returns all the predictions (empty list if no model given)