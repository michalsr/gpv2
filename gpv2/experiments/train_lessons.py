import json
import logging
import os
from argparse import ArgumentParser
from copy import deepcopy
from gpv2.data import lessons as lesson_data
import torch.utils.data
from transformers import AutoConfig
import io
from gpv2.data.lessons import *
from gpv2.data.dataset import Task
from gpv2.experiments.trainer_cli import *
from gpv2.image_featurizer.precomputed_features import Hdf5FeatureExtractor
from gpv2.model.gpv2 import T5GpvPerBox
from gpv2.model.layers import Linear, BasicBoxEmbedder
from gpv2.model.loss import DetrLocalizationLoss, LocalizationLoss, BasicGPVLoss
from gpv2.train.optimizer import DelayedWarmupScheduleBuilder, AdamWBuilder, OptimizerBuilder, \
  ParameterGroup, AllParameters
from gpv2.train.trainer import Trainer
from gpv2.utils import py_utils


from gpv2.utils.to_params import to_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--init_from")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--image_contrast",type=str,default=None)
  parser.add_argument("--text_contrast",type=str,default=None)
  parser.add_argument("--lesson",type=str,default=None)
  parser.add_argument("--mil",type=str,default=None)
  parser.add_argument("--synonym",type=str,default=None)

  add_train_args(
    parser, tasks=list(Task), epochs=8,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      # Hack the default to be t5-small if not manually specified
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])

  # Current best
  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=Hdf5FeatureExtractor("vinvl", BasicBoxEmbedder()),
    image_joiner=Linear(2048+5, t5_dim),
    all_lower_case=True,
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False,
  )

  groups = [ParameterGroup(
    AllParameters(),
    group_name="all",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  trainer: Trainer = get_trainer_from_args(
    args, logging_ema=0.995,
    optimizer=optimizer, scheduler=scheduler
  )
  if args.lesson != None:
    lesson_datasets = {'image_contrast':TrainerDataset(lesson_data.image_contrast.ImageContrastDataset('train'),"img-contrast"), "text_contrast":TrainerDataset(lesson_data.text_contrast.TextContrastDataset("train"),"text-contrast"),"mil":TrainerDataset(lesson_data.mil.MILDataset('train'),'mil'),
    'synonym':TrainerDataset(lesson_data.synonym.SynonymDataset("train"),"synonym-train")}

    training_lessons = []
    lesson_dict = {'img_contrast':args.image_contrast,'text_contrast':args.text_contrast,'mil':args.mil,'synonym':args.synonym}
    
    for lesson in lesson_dict:
      if lesson_dict[lesson] != None or args.lesson == 'all':
        training_lessons.append(lesson_datasets[lesson])
    if len(training_lessons) >1:
      for i,lesson_dataset in enumerate(training_lessons):
        if i == 0:
          trainer.upper_bound_no_change = 2 
          trainer.num_no_change_val = 0
          logging.info(f'Running lesson {i} out of {len(training_lessons)}')
          trainer.train_datasets.append(lesson_dataset)
          loc_setup = EvaluationSetup(
          evaluator.LocalizationEvaluator(),
          dict(beam_search_spec=None)
            )
          val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/val.json')
          trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
          trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
          trainer.stratify = True
          trainer.eval_loader = deepcopy(trainer.train_loader)
          trainer.train_loader.persist_workers = False
          trainer.eval_loader.persist_workers = False
          trainer.find_unused_parameters = args.find_unused

          run_trainer_from_args(trainer, model, args)
      else:
        trainer.upper_bound_no_change = 2 
        trainer.num_no_change_val = 0
        logging.info(f'Running lesson {i} out of {len(training_lessons)}')

        trainer.train_datasets = []
        trainer.train_datasets.append(lesson_dataset)
        loc_setup = EvaluationSetup(
          evaluator.LocalizationEvaluator(),
          dict(beam_search_spec=None)
            )
        val_file = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/val.json')
        trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_file),eval_setup=loc_setup))
        trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
        trainer.train_another_model(args.output_dir)
    else:
      trainer.upper_bound_no_change = 2 
      trainer.num_no_change_val = 0
      logging.info(f'Running single training lesson')
      trainer.train_datasets.append(training_lessons[0])
      loc_setup = EvaluationSetup(
      evaluator.LocalizationEvaluator(),
      dict(beam_search_spec=None)
        )
      val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/val.json')
      trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
      #trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=4857,eval_setup=loc_setup))
      trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
      trainer.stratify = True
      trainer.eval_loader = deepcopy(trainer.train_loader)
      trainer.train_loader.persist_workers = False
      trainer.eval_loader.persist_workers = False
      trainer.find_unused_parameters = args.find_unused

      run_trainer_from_args(trainer, model, args)




  run_trainer_from_args(trainer, model, args)


if __name__ == '__main__':
  main()
