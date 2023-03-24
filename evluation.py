#Evaluation metrics
def pixel_accuracy(prediction,truth):
  correct = (prediction==truth).sum().item()
  return correct/truth.numel()

def pixel_accuracy_for_class(prediction,truth,label):
  label_indices = truth == label
  correct = (prediction[label_indices] == truth[label_indices]).sum()
  return (correct/(label_indices).sum()).item()

def iou_for_class(prediction,truth,label):
  class_indices = truth == label
  intersect = (prediction[class_indices] == label).sum()
  union = class_indices.sum() + (prediction == label).sum() - intersect
  return (intersect/union).item()

def f1(prediction,truth,label):
  positive_indices = truth == label
  negative_indices = truth != label
  tp = (truth[positive_indices] == prediction[positive_indices]).sum().item()
  fp = (truth[negative_indices] != prediction[negative_indices]).sum().item()
  fn = (truth[positive_indices] != prediction[positive_indices]).sum().item()
  return (2*tp)/(2*tp + fp + fn)