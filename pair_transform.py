import random

#Transformation Applier
class FunctionalPairTransform:
  def __init__(self,transform,probability):
    self.transform = transform
    self.probability = probability

  def __call__(self,image,mask):
    if random.random() < self.probability:
      transformed_image = self.transform(image)
      transformed_mask = self.transform(mask)
      return transformed_image,transformed_mask
    else:
      return image,mask

class PairTransform:
  def __init__(self,random_transform,functional_transform,probability=1.0,random_keywords={},functional_keywords={}):
    self.random_transform = random_transform
    self.functional_transform = functional_transform
    self.random_keywords = random_keywords
    self.functional_keywords = functional_keywords
    self.probability = probability
  
  def __call__(self,image,mask):
    if random.random() < self.probability:
      args = self.random_transform.get_params(**self.random_keywords)
      transformed_image = self.functional_transform(image,*args,**self.functional_keywords)
      transformed_mask = self.functional_transform(mask,*args,**self.functional_keywords)
      return transformed_image,transformed_mask
    else:
      return image,mask