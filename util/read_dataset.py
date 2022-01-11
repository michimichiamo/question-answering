import pandas as pd
import random
import numpy as np

"""
Per come dobbiamo dare i dati alla NN cioè concatenando context e question, per il fatto che i prof vogliono come output un dict {id_question, testo}
per avere i dati comodi la lista è formata da:
[id, title, context, question, answer_start, answer_end]
in tale modo per il training i dati sono [context, question] e il target è [answer_start, answer_end]
"""

def read_dataset(path='training_dataset.json', validation_set_perc = 0.0, limit_dataset=0):
  '''
        @param path: path to dataset file
        @param validation_set_perc: inserire la percentuale in [0,1] del dataset per la validazione, se 0 non viene creato il validation set
        @param limit_dataset: if necessary to not use all the dataset set a limit of titles to include
  '''
  dataset = None
  with open(path) as file:
    dataset = pd.read_json(file)

  temp_dataset = []
  training_dataset = []
  validation_dataset = []

  titles = []
  i = 0
  for index, row in dataset.iterrows():
    #print(row['data'])
    data = row['data']
    title = data['title']
    titles.append(title)
    for ps in data['paragraphs']:
      #print(ps)
      context = ps['context']
      for qas in ps['qas']:
        id = qas['id']
        question = qas['question']
        for ans in qas['answers']:
          training_dataset.append([id, title, context, question, ans['answer_start'], ans['answer_start']+len(ans['text'])])
    if limit_dataset != 0 and i >= limit_dataset:
        break

  if validation_set_perc > 0.0:
    titles_validation = random.choices(titles, k=int(np.ceil(len(titles)*validation_set_perc/100)))
    for data in training_dataset:
      if data[1] in titles_validation:
        validation_dataset.append(data)
      else:
        temp_dataset.append(data)
    training_dataset = temp_dataset

  return training_dataset, validation_dataset
