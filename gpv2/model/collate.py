from collections import Callable
from typing import Any, List

from dataclasses import dataclass
from transformers import PreTrainedTokenizer

from gpv2.image_featurizer.image_featurizer import ImageCollater
from gpv2.model.gpv_example import GPVExample
import numpy as np
from gpv2.data.dataset import Task 
from gpv2.utils import py_utils

@dataclass
class CollateWithTokenizer(Callable):
  """Collate GPVExamples into tensor features"""

  tokenizer: PreTrainedTokenizer

  """How to collate the images"""
  image_collater: ImageCollater
  q_len: int
  ans_len: int

  """Extra collation to perform"""
  other_collate: Any = None

  def __call__(self, batch: List[GPVExample]):
    queries = []
    answers = []
    indicies = []
    mil_answers = []
    #print(type(batch[0]),'batch type')
    if type(batch[0]) == list:

      #print(batch[0],'batc 0')
    
      #print(batch[0],'batch')
      new_batch = py_utils.flatten_list(batch)

      # if batch[0][0].task == (Task.IMAGECONTRAST or Task.TEXTCONTRAST or Task.SYNONYM):
      #   new_batch = [item for sublist in batch for item in sublist]
   
      # if batch[0][0].task == Task.SYNONYM:
      
      #   for i in range(len(batch)):
      #     new_batch.append(batch[i][0])
      #     new_batch.append(batch[i][1])
      #     print(new_batch[-1].image_id,'1')
      #     print(new_batch[-2].image_id,'2')
      batch = new_batch
     
      if batch[0].task == Task.IMAGECONTRAST or batch[0].task == Task.TEXTCONTRAST:
        idx = batch[0].index_of_class
        idxes = [int(x.index_of_class) for x in batch]
        for y in batch:
          if int(y.index_of_class) != int(idx):
            print(idxes,'idxes')
            raise ValueError
        print('Images check out')
      if batch[0].task == (Task.SYNONYM):
        #image id should be the same for every pair 
        for i in range(0,len(batch),2):
          if str(batch[i].image_id) != str(batch[i+1].image_id):
            print(batch[i].image_id,batch[i+1].image_id)
            raise ValueError 
        # print(int(idx)==int(batch[1].index_of_class))
        # if not all(idxes) == int(idx):
        #   print(idxes)
        #   raise Error
     
      # for i,b in enumerate(new_batch):
      #   print(i,b.index_of_class,'index of class')
    # for i,ex in enumerate(batch):
    #   if i!= len(batch)-2:
    #     print(batch[i].image_id,batch[i+1].image_id)
    #     if str(batch[i].image_id) != str(batch[i+1].image_id):
    #       print('BAD')
    #       break
    # for i in range(len(batch)):
    #   if i<len(batch)-3:
    #     print(batch[i].image_id,'1 again')
    #     print(batch[i+1].image_id,'2 again')
    #print(batch[0].task,'batch task')
    print(len(batch),'batch size')
    for ex in batch:
      if ex.correct_answer!= None:
        mil_answers.append(ex.correct_answer)
      #print('Appended indicies')
      indicies.append(ex.index_of_class)
      if isinstance(ex.query, str):
        q = ex.query
      else:
        q = ex.query[np.random.randint(0, len(ex.query))]

      if ex.target_text is None or len(ex.target_text) == 0:
        # This is a bit messy since it conflates no output text requested (therefore, a
        # detection examples) with an unlabelled example (predicting a caption with no known label),
        # although there is no harm done since we ignore the labels when predicting anyway
        a = self.tokenizer.pad_token
      elif isinstance(ex.target_text, list):
        a = ex.target_text[np.random.randint(0, len(ex.target_text))]
      else:
        a = ex.target_text

      queries.append(q)
      answers.append(a)

    image_data = self.image_collater.collate(batch)
    image_inputs, box_targets = image_data

    queries = self.tokenizer(
      queries, return_tensors='pt', padding=True, max_length=self.q_len, truncation=True)
    answers = self.tokenizer(
      answers, return_tensors='pt', padding=True, max_length=self.ans_len)

    labels = dict(
      text_labels=answers["input_ids"],
      box_targets=box_targets,
      # Noted for logging purposes
      loss_logging=[None if x.meta is None else x.meta.get("loss-logging") for x in batch]
    )

    out = dict(
      input_ids=queries["input_ids"],
      input_mask=queries["attention_mask"],
      labels=labels,
      image_inputs=image_inputs
    )

    if self.other_collate:
      out.update(self.other_collate.collate(batch, out))
    return out
