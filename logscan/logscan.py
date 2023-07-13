
from . import __version__

# Libs
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import operator
import numpy as np

nltk.download('punkt')
regex_tokenizer = RegexpTokenizer(r'\w+')

from .auxiliares import is_word, replace_space, has_numbers, word_position, word_counter, remove_repeated
# Funções auxiliares

def log_template(cluster_tagger, log):
  new_log = replace_space(log)
  tokens = wordpunct_tokenize(new_log)
  variables_list = []
  template = ''
  for token in tokens:
    if token != "_IS_SPACE_":
      for word in cluster_tagger:
        if word[0] == token:
          info = word[3]
          break
      if info == 'variable':
        variables_list.append(token)
        template = template + '<*>'
      else:
        template = template + token
    else:
      template = template + ' '
  return template, variables_list


def word_classifier(wordfrequency):
  frequency_list = list(map(operator.itemgetter(2), wordfrequency))
  p30 = np.percentile(frequency_list, 30)
  label = []
  for word in wordfrequency:
    if word[2] < p30 or has_numbers(word[0]):
      # word[0] -> token
      # word[1] -> posicao
      # word[2] -> quantidade
      label.append((word[0],word[1],word[2],"variable"))
    else:
      label. append((word[0],word[1],word[2],"template"))
  return label

# Logscan

class LogScan:
  def __init__(self, logdata: list, header: bool, header_regex = None):
    print('- Logscan v1.0')
    if header:
      print('-- Header Extraction')
      loglist= [re.sub(f'{header_regex}', '', log) for log in logdata]
      self.data = pd.DataFrame(loglist,columns=['Log'])
    else:
      self.data = pd.DataFrame(logdata,columns=['Log'])

  def clean_data(self):
    print('-- Data Cleaning')
    clear_content = []
    for _, row in self.data.iterrows():
        raw_log = row['Log']
        log_tokens = regex_tokenizer.tokenize(raw_log)
        clean_text = []
        for token in log_tokens:
            if is_word(token):
                clean_text.append(token)
        clean_log = ' '.join(clean_text)
        clear_content.append(clean_log)
    self.data['CleanLog'] = clear_content

  def tfidf_transformer(self):
    print('-- TF-IDF Transformer')
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(self.data['CleanLog'])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    logs_embedding_df = pd.DataFrame(denselist, columns=feature_names)
    return logs_embedding_df

  def dbscanModel(self, logs_embedding_df):
    print('-- DBSCAN')
    clusterModel = DBSCAN(min_samples=2)
    clusterModel.fit(logs_embedding_df)
    self.data['Cluster'] = clusterModel.labels_

  def word_tagger(self):
    print('-- Word Tagger')
    tagger = []
    for cluster in np.unique(self.data['Cluster']):
      cluster_tokens = []
      dados_cluster = self.data.loc[self.data['Cluster'] == cluster]['Log']
      for log in dados_cluster:
        tokens = wordpunct_tokenize(log)
        tokens_position = word_position (tokens)
        cluster_tokens = cluster_tokens + tokens_position
      word_frequency = word_counter(cluster_tokens)
      new_wordfrequency = remove_repeated(word_frequency)
      wordlabel = word_classifier(new_wordfrequency)
      tagger.append((cluster, wordlabel))
    return tagger

  def create_templates(self, tagger):
    print('-- Template Extraction')
    templates = []
    variables = []
    for index, row in self.data.iterrows():
      log_cluster = row['Cluster']
      for cluster_tagger in tagger:
        if cluster_tagger[0] == log_cluster:
          log_tagger = cluster_tagger[1]
          break
      template, variables_list = log_template(log_tagger, row['Log'])
      templates.append(template)
      variables.append(variables_list)
    self.data['Template'] = templates
    self.data['Variables'] = variables

  def pipeline(self):
    self.clean_data()
    log_embedding_df = self.tfidf_transformer()
    self.dbscanModel(log_embedding_df)
    tagger = self.word_tagger()
    self.create_templates(tagger)
    return tagger, self.data


def main():
  print('Logscan')
  print(f"The package's version is: {__version__}")

  # Teste
  android_dataset = pd.read_csv("logs\Andriod_2k.log_structured.csv")
  log_scan_android = LogScan(list(android_dataset['Content']), header=False)
  tagger_android, result_dataset = log_scan_android.pipeline()
  result_dataset.to_csv('resultados.csv')