import math
from collections import Counter

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english")

ner_task = pipeline(task='ner', model=model, tokenizer=tokenizer)

class Entity_Probabilities(Counter):
    """
    A Counter with softmax.
    We use this to get a probability distribution over the company entities.
    """    
    def softmax(self):
      for key in self:
        self[key] = math.exp(self[key])
      total = float(sum(self.values()))
      for key in self:
          self[key] /= total
      return self


def get_main_issuer_entites(text: str):
  """
  This function outputs a probability distribution over the company entities found in 'text'.
  """
  results = ner_task(text)
  issuers = []
  prev_index = 0
  # BertTokenizer sometimes splits words into smaller meaningful units. They have to be joined back together.
  for index, token in enumerate(results):
    # label-6 corresponds to ORG entity. Its just an error that's yet to be corrected in the package.
    if token['entity'] == "LABEL_6":
      if (index == prev_index+1) and  (prev_index!=0):
        f = lambda x: x.strip('#') if x.startswith("#") else " "+x
        entity_token = f(token['word'])
        issuers[-1] = (index , issuers[-1][1] + entity_token)
        prev_index = index
      else:
        issuers.append((index, token['word']))
        prev_index = index
  _, issuer_list = zip(*issuers)
  # we apply softmax to the count of "issuer entities" in the text and return in the order of probabilities.
  return Entity_Probabilities(issuer_list).softmax().most_common()