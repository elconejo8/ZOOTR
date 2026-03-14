import cv2
import os
import torch
from torch import softmax
from PIL import Image
import time
from decord import VideoReader, cpu


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_pred(img, model, transform = None):

    if transform is not None:
        img = transform(img)

    img = img.to(device)

    out = model(img[None, :, :, :])
    pred = softmax(out, 1).cpu().detach().numpy()[0]
    return pred[1].item()
#Get prediction on a single image
    


def save_video_frames(vid, save_to_path, n,  model=None, transform=None):
    model = model.to(device)
    model.eval()
    start = time.time()
    video_capture = cv2.VideoCapture(vid)
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions =  []
    overhead = time.time() 
    capture_times = []
    colour_times = []
    model_times = []
    for j in range(v_len):
        pre_vid_capture = time.time()
        success, vframe = video_capture.read()
        post_vid_capture = time.time()
        capture_times += [post_vid_capture - pre_vid_capture]
        if j % n == 0:
            pre_colour = time.time()
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            post_colour = time.time()
            colour_times += [post_colour - pre_colour]
            pre_model_time = time.time()
            if model is not None:
                pred = get_image_pred(vframe, model, transform)
                predictions.append(float(pred))
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '-' + str(round(float(pred), 2)) + '.png'))
            else:
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '.png'))
            post_model_time = time.time()
            model_times += [post_model_time - pre_model_time]
        if j >= 1000:
            break
    return predictions, {'overhead': overhead, 'capture_times': capture_times, 'colour_times': colour_times, 'model_times': model_times, 'total_time': time.time() - start}
#Gets a video and saves every n frame to specified folder. Frame name is it's sequential number. 
#If given a model it calculates probability of frame being an item and appends it to the name.
#Returns all the predictions (empty list if no model given)


def batch_frame_predict(video_location, model, transform, device, batch_size=20):
    vr = VideoReader(video_location, ctx=cpu(0)) 
    total_frames = len(vr)
    all_predictions = []

    for i in range(0, total_frames, batch_size):
        batch_idx = vr[i:i+batch_size]
        x_tensor = torch.from_numpy(batch_idx.asnumpy()) 
        batch_tensor = transform(x_tensor)
        with torch.no_grad():
            outputs = model(batch_tensor.to(device))
            probs = softmax(outputs, 1).cpu().numpy()[:, 1]
            all_predictions += list(probs)
    
    return all_predictions