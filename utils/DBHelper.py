import datetime

import peewee
from peewee import *

from utils.ProjectConfig import ProjectConfig
from utils.mysql_config import *

db = MySQLDatabase('experiment_data', host=HOST, port=PORT, user=USER, passwd=PASSWORD)

class Record(peewee.Model):
  id = peewee.AutoField(primary_key=True)
  baseline = peewee.BooleanField()
  model = peewee.TextField()
  dataset = peewee.TextField()
  train_size = peewee.TextField()
  n_aug = peewee.IntegerField()
  f1 = peewee.DecimalField(11,10)
  accuracy = peewee.DecimalField(11,10)
  epochs = peewee.IntegerField()
  repeat_id = peewee.IntegerField()
  augmentation = peewee.TextField(null=True)
  aug_prob = peewee.DecimalField(6, 5, null=True)
  date = peewee.DateField()
  datetime = peewee.DateTimeField()
  run_time = peewee.TextField()
  seed = peewee.IntegerField()

  class Meta:
    database = db

db.connect()
db.create_tables([Record])

def save_result(task_config, repeat_id, f1, accuracy, run_time, seed):
  record = Record(baseline=task_config['baseline'],
                  model=task_config['model'],
                  dataset=task_config['dataset'],
                  train_size=task_config['train_size'],
                  n_aug=task_config['n_aug'],
                  f1=f1, accuracy=accuracy,
                  epochs=task_config['epochs'],
                  repeat_id=repeat_id,
                  augmentation=task_config['augmentations'][0] if len(task_config['augmentations']) else None,
                  aug_prob=task_config['aug_params'][0] if len(task_config['aug_params']) else None,
                  date=datetime.datetime.now().strftime('%Y-%m-%d'),
                  datetime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  run_time=run_time,
                  seed=seed)
  record.save()