import cv2
import os
import torch
from torch import softmax
from PIL import Image
from decord import VideoReader, cpu


def get_image_pred(img, model, device, transform = None):

    if transform is not None:
        img = transform(img)

    img = img.to(device)

    out = model(img[None, :, :, :])
    pred = softmax(out, 1).cpu().detach().numpy()[0]
    return pred[1].item()
#Get prediction on a single image
    


def save_video_frames(vid, save_to_path, n,  device, model=None, transform=None):
    model = model.to(device)
    model.eval()

    video_capture = cv2.VideoCapture(vid)
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions =  []

    for j in range(v_len):
        _, vframe = video_capture.read()
        if j % n == 0:
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            if model is not None:
                pred = get_image_pred(vframe, model, transform)
                predictions.append(float(pred))
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '-' + str(round(float(pred), 2)) + '.png'))
            else:
                Image.fromarray(vframe).save(os.path.join(save_to_path, str(j) + '.png'))

    return predictions
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


def see_frames(video_location, item_frames):
    vr = VideoReader(video_location, ctx=cpu(0))

    chosen_images = vr.get_batch(item_frames).asnumpy()

    for img in chosen_images:
        Image.fromarray(img).show()