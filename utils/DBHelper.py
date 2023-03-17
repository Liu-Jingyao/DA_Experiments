import datetime

import peewee
from peewee import *

from utils.ProjectConfig import ProjectConfig

db = MySQLDatabase()

class Record(peewee.Model):
  id = peewee.AutoField(primary_key=True)
  baseline = peewee.BooleanField()
  model = peewee.TextField()
  dataset = peewee.TextField()
  f1 = peewee.DecimalField(11,10)
  accuracy = peewee.DecimalField(11,10)
  epochs = peewee.IntegerField()
  repeat_id = peewee.IntegerField()
  augmentation = peewee.TextField()
  aug_prob = peewee.DecimalField(6,5)
  date = peewee.DateTimeField(default=datetime.datetime.now().strftime('%Y-%m-%d'))

  class Meta:
    database = db

db.connect()
db.create_tables([Record])

def save_result(task_config, repeat_id, f1, accuracy):
  record = Record(baseline=task_config['baseline'],
                  model=task_config['model'],
                  dataset=task_config['dataset'],
                  f1=f1, accuracy=accuracy,
                  epochs=task_config['epochs'],
                  repeat_id=repeat_id,
                  augmentation=task_config['augmentations'][0] if len(task_config['augmentations']) else None,
                  aug_prob=task_config['aug_params'][0] if len(task_config['aug_params']) else None)
  record.save()
