import numpy as np
import torch

INDENT = " " * 4

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

def true_positive_all_positive_classwise(prediction,truth,label):
  label_indices = truth == label
  correct = (prediction[label_indices] == truth[label_indices]).sum()
  total = label_indices.sum()
  if total == 0:
    return None
  return correct.item(),total.item()

def tp_ap_all_classes(prediction,truth,n_labels):
  return [true_positive_all_positive_classwise(prediction,truth,label) for label in range(n_labels)]

def pixel_accuracy_all_classes(prediction,truth,n_labels):
  return [pixel_accuracy_for_class(prediction,truth,label) for label in range(n_labels)]

def iou_for_class(prediction,truth,label):
  class_indices = truth == label
  intersect = (prediction[class_indices] == label).sum()
  union = class_indices.sum() + (prediction == label).sum() - intersect
  if union == 0:
    return None
  return (intersect/union).item()

def intersect_union_classwise(prediction,truth,label):
  class_indices = truth == label
  intersect = (prediction[class_indices] == label).sum()
  union = class_indices.sum() + (prediction == label).sum() - intersect
  if union == 0:
    return None
  return intersect.item(),union.item()

def in_union_all_classes(prediction,truth,n_labels):
  return [intersect_union_classwise(prediction,truth,label) for label in range(n_labels)]

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

def tp_fp_fn_classwise(prediction,truth,label):
  positive_indices = truth == label
  negative_indices = truth != label
  tp = (truth[positive_indices] == prediction[positive_indices]).sum().item()
  fp = (truth[negative_indices] != prediction[negative_indices]).sum().item()
  fn = (truth[positive_indices] != prediction[positive_indices]).sum().item()
  denominator = 2*tp + fp + fn
  if denominator == 0:
    return None
  return tp,fp,fn

def confusions(prediction,truth,n_labels):
  class_confusions = np.zeros((n_labels,n_labels))
  for truth_label in range(n_labels):
    label_indices = truth == truth_label
    for prediction_label in range(n_labels):
      n_confusions = (prediction[label_indices] == prediction_label).sum().item()
      class_confusions[truth_label,prediction_label] += n_confusions
  return class_confusions

def tp_fp_fn_all_classes(prediction,truth,n_labels):
  return [tp_fp_fn_classwise(prediction,truth,label) for label in range(n_labels)]

def f1_all_classes(prediction,truth,n_labels):
  return [f1(prediction,truth,label) for label in range(n_labels)]


class MetricManager():
  #All metric functions must take same arguments

  def __init__(self,labels):
    self.functions = {}
    self.function_keywords = {}
    self.stats = {}
    self.labels = labels
    self.n = {}

  def add_metric(self,name,function,record_n=False,**keywords):
    self.functions[name] = function
    self.function_keywords[name] = keywords
    self.stats[name] = []
    if record_n:
      self.n[name] = []

  def apply_metrics(self,prediction,truth):
    label_pred = torch.argmax(prediction,1)
    label_truth = torch.argmax(truth,1) 
    for name in self.functions.keys():
      result = self.functions[name](
        label_pred,label_truth,**self.function_keywords[name])
      if name in self.n:
        self.stats[name].append(result[0])
        self.n[name].append(result[1])
      else:
        self.stats[name].append(result)

  def get_metric(self,name):
    return np.array(self.stats[name])
  
  #Pretty print specific metric result 
  def pp_metric(self,name,index):
    data = self.stats[name][index]
    if not isinstance(data,list):
      return data
    
    return "\n"+"\n".join([f"{INDENT}{l}: {d}" for d,l in zip(data,self.labels)])