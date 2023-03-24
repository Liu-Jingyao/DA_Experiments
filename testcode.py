import datetime
import os
import random

import nltk
import peewee
import yaml
from peewee import MySQLDatabase

from data_augmentations.EDA import synonym_replacement
from utils.ProjectConfig import ProjectConfig
from utils.consts import ROOT_PATH
from utils.data_utils import WordClean

import pathlib

db = MySQLDatabase('experiment_data', host='82.157.53.138', port=3306, user='experiment_data', passwd='A54001769a')

class Record(peewee.Model):
  id = peewee.IntegerField()
  baseline = peewee.BooleanField()
  model = peewee.TextField()
  dataset = peewee.TextField()
  f1 = peewee.DecimalField()
  accuracy = peewee.DecimalField()
  epoch = peewee.IntegerField()
  repeat_id = peewee.IntegerField()
  augmentation = peewee.TextField()
  aug_prob = peewee.DecimalField()
  date = peewee.DateTimeField(default=datetime.datetime.now().strftime('%Y-%m-%d'))

  class Meta:
    database = db

if __name__ == '__main__':
    db.connect()
    db.create_tables([Record])
    record = Record(baseline=True, model='b', accuracy=0.5)
    record.save()