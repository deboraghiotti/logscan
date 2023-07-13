import re
import numpy as np

def is_word(inputString):
  return bool(re.search(r'^[a-zA-Z]+$', inputString))

def replace_space(old_string):
  new_string = old_string.replace(" ", " _IS_SPACE_ ")
  return new_string

def has_numbers(inputString):
  return bool(re.search(r'\d', inputString))

def word_position (wordlist):
  word_position_list = []
  position = 0
  for word in wordlist:
    word_position_list.append((word, position))
    position = position + 1
  return word_position_list

def word_counter(wordlist):
  wordfreq = []
  listw = wordlist.copy()
  word_frequency = []
  for w in listw:
    frequency = listw.count(w)
    wordfreq.append(frequency)
    word_frequency.append((w[0], w[1], frequency))
  return word_frequency

def remove_repeated(wordfrequency):
  new_wordfrequency = []
  for word in wordfrequency:
    if word not in new_wordfrequency:
      new_wordfrequency.append(word)
  return new_wordfrequency

# Funcoes de avaliacao

def parsing_accuracy(data):
  log_per_template =  data['EventId'].value_counts().to_dict()
  correct = 0
  for cluster in np.unique(data['Cluster']):
    data_cluster = data.loc[data['Cluster'] == cluster]
    log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
    for eventid in np.unique(data_cluster['EventId']):
      if log_per_template[eventid] == log_per_template_cluster[eventid]:
        correct = correct + log_per_template_cluster[eventid]
  return correct/len(data)

def cluster_accuracy(data):
  # CA - 
  correct = 0
  for cluster in np.unique(data['Cluster']):
    data_cluster = data.loc[data['Cluster'] == cluster]
    log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
    for eventid in np.unique(data_cluster['EventId']):
      if len(np.unique(data_cluster['EventId'])) == 1:
        correct = correct + log_per_template_cluster[eventid]
  return correct/len(data)

def parsing_cluster_accuracy(data):
  # Parsing Accuracy + Cluster Accuracy
  log_per_template =  data['EventId'].value_counts().to_dict()
  correct = 0
  for cluster in np.unique(data['Cluster']):
    data_cluster = data.loc[data['Cluster'] == cluster]
    log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
    for eventid in np.unique(data_cluster['EventId']):
      if log_per_template[eventid] == log_per_template_cluster[eventid] and len(np.unique(data_cluster['EventId'])) == 1:
        correct = correct + log_per_template_cluster[eventid]
  return correct/len(data)

def cluster_evaluation(data):
  resultado1 = parsing_accuracy(data)
  resultado2 = cluster_accuracy(data)
  resultado3 = parsing_cluster_accuracy(data)
  resultado = (resultado1 + resultado2 + resultado3)/3
  return resultado, resultado1, resultado2, resultado3