import numpy as np
import torch

def calculate_class_weights(train_set,n_labels,device="cpu"):
  #Determine no. pixels of each class in training set
  n_pixels = np.zeros(n_labels)
  for _,mask in train_set:
    pixel_labels = torch.argmax(mask,0)
    for label in range(n_labels):
      n_pixels[label] += (pixel_labels==label).sum().item()

  total_pixels = np.sum(n_pixels)
  weights = total_pixels / n_pixels
  return torch.Tensor(weights / np.sum(weights)).to(device)