import datetime

import peewee
from peewee import *

from utils.ProjectConfig import ProjectConfig

db = MySQLDatabase('experiment_data', host='82.157.53.138', port=3306, user='root', passwd='789556')

class Record(peewee.Model):
  id = peewee.IntegerField()
  baseline = peewee.BooleanField()
  model = peewee.TextField()
  dataset = peewee.TextField()
  f1 = peewee.DecimalField()
  accuracy = peewee.DecimalField()
  epochs = peewee.IntegerField()
  repeat_id = peewee.IntegerField()
  augmentation = peewee.TextField()
  aug_prob = peewee.DecimalField()
  date = peewee.DateTimeField(default=datetime.datetime.now().strftime('%Y-%m-%d'))

  class Meta:
    database = db

def save_result(task_config, baseline, repeat_id, f1, accuracy):
  record = Record(baseline=baseline,
                  model=task_config['model'],
                  dataset=task_config['datasset'],
                  f1=f1, accuracy=accuracy,
                  epochs=task_config['epochs'],
                  repeat_id=repeat_id,
                  augmentation=task_config['augmentations'][0],
                  aug_prob=task_config['aug_params'][0])
  record.save()