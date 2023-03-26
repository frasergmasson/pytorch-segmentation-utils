import torch
import numpy as np

def mask_to_img(mask,label_colours):
  mask = torch.argmax(mask,0)
  mask = mask.cpu().detach().numpy()
  img = np.zeros((mask.shape[0],mask.shape[1],3))
  for y in range(mask.shape[0]):
    for x in range(mask.shape[1]):
      pixel = mask[y,x]
      img[y,x,:] = label_colours[pixel]
  return img