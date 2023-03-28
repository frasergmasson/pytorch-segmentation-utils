import numpy as np
import torch

#Evaluation metrics
def pixel_accuracy(prediction,truth):
  correct = (prediction==truth).sum().item()
  return correct/truth.numel()

def pixel_accuracy_for_class(prediction,truth,label):
  label_indices = truth == label
  correct = (prediction[label_indices] == truth[label_indices]).sum()
  total = label_indices.sum()
  if total == 0:
    return None
  return (correct/total).item()

def pixel_accuracy_all_classes(prediction,truth,n_labels):
  return [pixel_accuracy_for_class(prediction,truth,label) for label in range(n_labels)]

def iou_for_class(prediction,truth,label):
  class_indices = truth == label
  intersect = (prediction[class_indices] == label).sum()
  union = class_indices.sum() + (prediction == label).sum() - intersect
  if union == 0:
    return None
  return (intersect/union).item()

def iou_all_classes(prediction,truth,n_labels):
  return [iou_for_class(prediction,truth,label) for label in range(n_labels)]

def f1(prediction,truth,label):
  positive_indices = truth == label
  negative_indices = truth != label
  tp = (truth[positive_indices] == prediction[positive_indices]).sum().item()
  fp = (truth[negative_indices] != prediction[negative_indices]).sum().item()
  fn = (truth[positive_indices] != prediction[positive_indices]).sum().item()
  denominator = 2*tp + fp + fn
  if denominator == 0:
    return None
  return (2*tp)/denominator

def f1_all_classes(prediction,truth,n_labels):
  return [f1(prediction,truth,label) for label in range(n_labels)]


class MetricManager():
  #All metric functions must take same arguments

  def __init__(self):
    self.functions = {}
    self.function_keywords = {}
    self.stats = {}

  def add_metric(self,name,function,**keywords):
    self.functions[name] = function
    self.function_keywords[name] = keywords
    self.stats[name] = []

  def crunch(self,prediction,truth):
    label_pred = torch.argmax(prediction,1)
    label_truth = torch.argmax(truth,1) 
    for name in self.functions.keys():
      result = self.functions[name](label_pred,label_truth,**self.function_keywords[name])
      self.stats[name].append(result)

  def get_metric(self,name):
    return np.array(self.stats[name])