"""A flask server which renders test results."""
import os
import sys
import threading
import pathlib
import glob
import random
import flask
import flask_sqlalchemy
import portpicker
import shutil
import sqlalchemy as sql

from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.samplers import validation_database
from deeplearning.benchpress.samplers import samples_database
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.dashboard import dashboard_db
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.proto import internal_pb2
from absl import flags
import humanize

from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

# Disable flask banner on load.
_cli = sys.modules["flask.cli"]
_cli.show_server_banner = lambda *x: None

flags.DEFINE_integer(
  "clgen_dashboard_port", None, "The port to launch the server on.",
)

MEDIA_PATH = pathlib.Path(environment.DASHBOARD_STATIC, "images")
MEDIA_PATH.mkdir(exist_ok = True)
flask_app = flask.Flask(
  __name__,
  template_folder = environment.DASHBOARD_TEMPLATES,
  static_folder = environment.DASHBOARD_STATIC,
)

db = flask_sqlalchemy.SQLAlchemy(flask_app)

data = {}
cached_models = {}
cached_corpuses = {}
cached_samplers = {}

def GetBaseTemplateArgs():
  l.logger().debug("deeplearning.clgen.dashboard.GetBaseTemplateArgs()")
  return {
    "urls": {
      "cache_tag": str(5),
      "site_css": flask.url_for("static", filename="site.css"),
      "site_js": flask.url_for("static", filename="site.js"),
    },
    "build_info": {
      "html": "Description",
      "version": 2,
    },
  }

def parseCorpus(workspace_path):

  corpuses = []
  if (workspace_path / "corpus" / "encoded").exists():
    corpus_path = workspace_path / "corpus" / "encoded"
    for corpus_sha in corpus_path.iterdir():
      encoded_db = encoded.EncodedContentFiles("sqlite:///{}".format(corpus_sha / "encoded.db"), must_exist = True)
      corpus = {
        'path': corpus_path / corpus_sha,
        'sha' : str(corpus_sha.stem),
        'datapoint_count': encoded_db.size,
        'summary': "{} datapoint corpus, {}".format(encoded_db.size, str(corpus_sha.stem)),
        'models' : parseModels(workspace_path, str(corpus_sha.stem))
      }
      global cached_corpuses
      cached_corpuses[crypto.sha256_str(str(workspace_path.name) + str(corpus_sha.name))] = corpus
      corpuses.append(corpus)
  return corpuses

def parseModels(workspace_path, corpus_sha: str):

  models = []
  if (workspace_path / "model").exists():
    for model_sha in (workspace_path / "model").iterdir():
      model_path = workspace_path / "model" / model_sha
      if (model_path / "tokenizer").exists() and pathlib.Path(os.readlink(model_path / "tokenizer")).parent.name == corpus_sha:
        if (model_path / "META.pbtxt").exists():
          meta = parseMeta(model_path / "META.pbtxt")
          model = {          
            'path'        : model_path,
            'sha'         : str(model_sha.name),
            'config'      : meta,
            'tokenizer'    : tokenizers.TokenizerBase.FromFile(model_path / pathlib.Path(os.readlink(model_path / "tokenizer"))),
            'training_log': parseTrainLogs(model_path / "logs"), # TODO
            'validation'  : parseValidationDB(model_path / "logs" / "validation_samples.db"),
            'samplers'    : parseSamplers(workspace_path, model_path / "samples", str(model_sha.name)), # TODO sample_db ?
            'summary'     : parseModelSummary(meta)
          }
          global cached_models
          cached_models[crypto.sha256_str(str(workspace_path.name) + str(model_sha.name))] = model
          models.append(model)


  return models

def parseMeta(meta):
  with open(meta, 'r') as f:
    return f.read().splitlines()

def parseModelSummary(meta):
  m = pbutil.FromString('\n'.join(meta), internal_pb2.ModelMeta())
  if m.config.architecture.backend == model_pb2.NetworkArchitecture.TENSORFLOW_BERT:
    summary = ("BERT, hs: {}, nhl: {}, atth: {}, imsz: {}, pemb: {}, preds: {}, dp: {}, mprob: {}, {}"
        .format(m.config.architecture.hidden_size, m.config.architecture.num_hidden_layers, m.config.architecture.num_attention_heads,
        m.config.architecture.intermediate_size, m.config.architecture.max_position_embeddings, m.config.training.max_predictions_per_seq, 
        m.config.training.dupe_factor, round(m.config.training.masked_lm_prob, 3),
        "mask" if m.config.training.data_generator.HasField("mask") else 
            "hole-{},{}".format(
                (m.config.training.data_generator.hole.absolute
                if m.config.training.data_generator.hole.HasField("absolute_length")
                else m.config.training.data_generator.hole.relative_length),
                "unf" if m.config.training.data_generator.hole.HasField("uniform_distribution") else
                  "norm-{},{}".format(
                    round(m.config.training.data_generator.hole.normal_distribution.mean, 2), 
                    round(m.config.training.data_generator.hole.normal_distribution.variance, 2)
                    )
                )
        )
      )
  else:
    raise NotImplementedError
  return summary

def parseSamplerSummary(meta):
  m = pbutil.FromString('\n'.join(meta), internal_pb2.SamplerMeta())
  summary = ("Sequence length: {}, temperature: {}".format(
      m.config.sequence_length, 
      m.config.temperature_micros / 1e6,
    )
  )
  return summary

def parseTrainLogs(logs):

  log_tensors = {}
  if len(glob.glob(str(logs / "events.out.tfevents*"))) != 0:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(str(logs))
    event_acc.Reload()
    if 'scalars' in event_acc.Tags():
      for tag in event_acc.Tags()['scalars']:
        wall_time, steps, value = zip(*event_acc.Scalars(tag))
        log_tensors[tag] = [{'wall_time': w, 'step_num': s, 'value': v} for w, s, v in zip(wall_time, steps, value)]
  return log_tensors

def parseValidationDB(db_path):

  validation_db = {
    'val_sample_count': -1,
    'path': None,
    'val_metrics': [],
    'val_samples': [],
  }
  try:
    if db_path.exists():
      validation_db['path'] = "sqlite:///{}".format(db_path)
      val_db = validation_database.ValidationDatabase(validation_db['path'], must_exist = True)
      validation_db['val_sample_count'] = val_db.count
  except:
    validation_db['val_sample_count'] = -1
    validation_db['path'] = None
  return validation_db

def parseSamplers(workspace_path, sample_path, model_sha):

  global cached_samplers
  model_samplers = []

  if sample_path.exists():
    for sampler_sha in sample_path.iterdir():
      if ((workspace_path / "sampler" / sampler_sha.name / "META.pbtxt").exists() and
          (workspace_path / "sampler" / sampler_sha.name / "samples" / model_sha).exists()):
        meta = parseMeta(str(workspace_path / "sampler" / sampler_sha.name / "META.pbtxt"))
        path = (workspace_path / "sampler" / sampler_sha.name / "samples" / model_sha)
        sample_dbs = {}
        for db in path.iterdir():
          if db.suffix == ".db":
            sample_dbs[db.stem] = db
        sampler = {
          'path': path,
          'sha': sampler_sha.name,
          'config': meta,
          'summary': parseSamplerSummary(meta),
          'sample_dbs': sample_dbs,
        }
        cached_samplers[sampler_sha.name] = sampler
        model_samplers.append(sampler)
  return model_samplers

def parseData():

  dashboard_path = pathlib.Path(FLAGS.workspace_dir).absolute()
  workspaces = [p for p in dashboard_path.iterdir() if p.is_dir()]
  global data
  data = {
    "workspaces": {
      p: {
        'name': p.stem,
        'path': p,
        'corpuses': parseCorpus(p),
        } for p in workspaces
    },
  }
  return data

@flask_app.route("/")
def index():
  global data
  data = parseData()
  return flask.render_template(
    "dashboard.html", data = data, **GetBaseTemplateArgs()
  )

@flask_app.route("/<string:workspace>/corpus/<string:corpus_sha>/")
def corpus(workspace: str, corpus_sha: str):
  global data
  global cached_corpuses
  if data == {}:
    data = parseData()

  target_sha = crypto.sha256_str(str(workspace) + corpus_sha)
  corpus = cached_corpuses[target_sha]
  corpus_stats = []

  for d in glob.glob(str(corpus['path'] / "*.png")):
    png_path = pathlib.Path(d)
    dest_file = MEDIA_PATH / workspace / corpus_sha / png_path.name
    dest_file.parent.mkdir(exist_ok = True, parents = True)
    shutil.copyfile(
      png_path,
      str(dest_file)
    )
    corpus_stats.append(
      {
        'name': png_path.stem,
        'plot':
          "/" + str(dest_file.relative_to(pathlib.Path(flask_app.static_folder).parent))
      }
    )
  corpus['stats'] = corpus_stats
  print(corpus['summary'])
  return flask.render_template("corpus.html", data = corpus, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/model_specs")
def model_specs(workspace: str, model_sha: str):
  global data
  global cached_models
  if data == {}:
    data = parseData()

  target_sha = crypto.sha256_str(str(workspace) + model_sha)
  current_model = cached_models[target_sha]
  spec_data ={
    'config': current_model['config']
  }
  return flask.render_template("model_specs.html", data = spec_data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/dataset")
def dataset(workspace: str, model_sha: str):
  global data
  global cached_models
  if data == {}:
    data = parseData()

  target_sha    = crypto.sha256_str(str(workspace) + model_sha)
  current_model = cached_models[target_sha]

  datasets = []
  for d in glob.glob(str(current_model['path'] / "dataset" / "*.png")):
    png_path = pathlib.Path(d)
    dest_file = MEDIA_PATH / workspace / model_sha / "dataset" / png_path.name
    dest_file.parent.mkdir(exist_ok = True, parents = True)
    shutil.copyfile(
      png_path,
      str(dest_file)
    )
    datasets.append(
      {
        'name': png_path.stem,
        'plot': 
          "/" + str(dest_file.relative_to(pathlib.Path(flask_app.static_folder).parent))
      }
    )
  spec_data = {
    'summary'  : current_model['summary'],
    'workspace': workspace,
    'model_sha': model_sha,
    'datasets' : datasets,
  }
  return flask.render_template("dataset.html", data = spec_data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/sampler/<string:sampler_sha>/sampler_specs")
def sampler_specs(workspace: str, sampler_sha: str):
  global data
  global cached_samplers
  if data == {}:
    data = parseData()

  current_sampler = cached_samplers[sampler_sha]
  for i, l in enumerate(current_sampler['config']):
    if 'start_text' in l:
      current_sampler['config'][i] = current_sampler['config'][i].replace("\\n", "\n")

  spec_data ={
    'config': current_sampler['config']
  }
  return flask.render_template("sampler_specs.html", data = spec_data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/validation")
def validation_samples(workspace: str, model_sha: str):
  global data
  global cached_models
  if data == {}:
    data = parseData()

  target_sha = crypto.sha256_str(str(workspace) + model_sha)
  current_model = cached_models[target_sha]
  validation = current_model['validation']

  if validation['path']:

    val_db = validation_database.ValidationDatabase(str(validation['path']), must_exist = True)
    with val_db.Session() as session:
      validation['val_samples'] = session.query(validation_database.BERTValFile).all()
      validation['val_metrics'] = session.query(validation_database.ValResults).all()
      # random.shuffle(validation['val_samples'])

    for sample in validation['val_samples']:
      processed_input_ids = []
      if '[HOLE]' in sample.input_ids:
        mask_type = '[HOLE]'
      elif '[MASK]' in sample.input_ids:
        mask_type = '[MASK]'
      else:
        mask_type = ''

      if mask_type == '[HOLE]':
        input_ids = sample.input_ids.split(mask_type)
        mask_num = sample.num_targets
        for i in range(mask_num):
          processed_input_ids += [
            {
              'text': input_ids[i],
              'color': 'plain',
              'length': len(input_ids[i]),
            },
            {
              'text': mask_type,
              'color': 'hole',
              'length': int(sample.masked_lm_lengths.split(',')[i]),
            },
            {
              'text': sample.masked_lm_predictions.split('\n')[i].replace(' ', '[ ]').replace('\n', '\\n'),
              'color': 'prediction',
              'length': 1,
            },
            {
              'text': sample.masked_lm_ids.split('\n')[i].replace(' ', '[ ]').replace('\n', '\\n'),
              'color': 'target',
              'length': 1,
            },
          ]
        while i < len(input_ids) - 1:
          i += 1
          processed_input_ids.append(
            {
              'text': input_ids[i],
              'color': 'plain',
              'length': len(input_ids[i]),
            },
          )

      elif mask_type == '[MASK]':
        processed_input_ids = [
          {
            'text': sample.input_ids,
            'color': 'plain',
          }
        ]

      sample.input_ids = processed_input_ids
  validation['summary']    = current_model['summary']
  validation['workspace']  = workspace
  validation['model_sha']  = model_sha
  return flask.render_template("validation_samples.html", data = validation, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/sampling")
def sampling(workspace: str, model_sha: str):
  global data
  global cached_models
  if data == {}:
    data = parseData()

  target_sha = crypto.sha256_str(str(workspace) + model_sha)
  current_model = cached_models[target_sha]
  samplers = current_model['samplers']

  data = {
    'summary'  : current_model['summary'],
    'workspace': workspace,
    'model_sha': model_sha,
    'samplers' : samplers,
  }
  return flask.render_template("sampling.html", data = data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/training")
def training(workspace: str, model_sha: str):
  global data
  global cached_models
  if data == {}:
    data = parseData()
  
  data['plots'] = []

  target_sha = crypto.sha256_str(str(workspace) + model_sha)
  for d in glob.glob(str(cached_models[target_sha]['path'] / "logs" / "*.png")):
    png_file = pathlib.Path(d)
    dest_file = MEDIA_PATH / workspace / model_sha / "logs" / png_file.name
    dest_file.parent.mkdir(exist_ok = True, parents = True)
    shutil.copyfile(png_file, dest_file)
    data['plots'].append(
      "/" + str(dest_file.relative_to(pathlib.Path(flask_app.static_folder).parent))
    )

  data['summary']   = cached_models[target_sha]['summary']
  data['workspace'] = workspace
  data['model_sha'] = model_sha
  return flask.render_template("training.html", data = data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/sampler/<string:sampler_sha>/<string:sample_db>")
def sample_files(workspace: str, model_sha: str, sampler_sha: str, sample_db: str):

  global data
  global cached_models
  if data == {}:
    data = parseData()

  current_sampler = {}
  target_sha = crypto.sha256_str(str(workspace) + model_sha)

  for sampler in cached_models[target_sha]['samplers']:
    if sampler['sha'] == sampler_sha:
      current_sampler = sampler
      break

  db_file = current_sampler['path'] / "{}.db".format(sample_db)
  samples_db = samples_database.SamplesDatabase("sqlite:///{}".format(db_file), must_exist = True)

  with samples_db.Session() as session:
    sample_files = session.query(samples_database.Sample).all()

  for sample in sample_files:
    processed_feed = []
    processed_indices = []
    if '[HOLE]' in sample.sample_feed:
      mask_type = '[HOLE]'
    elif '[MASK]' in sample.sample_feed:
      mask_type = '[MASK]'
    else:
      mask_type = ''
    sample_feed    = sample.sample_feed.split(mask_type)
    sample_indices = sample.sample_indices.split('\n')
    assert len(sample_feed) - 1 == len(sample_indices), ("sample hole length/generation mismatch: {}, {}"
            .format(
              len(sample_feed),
              len(sample_indices),
              )
            )

            
    prediction = sample.text

    for i in range(len(sample_feed) - 1):
      processed_feed += [
        {
          'text' : sample_feed[i],
          'color': 'plain',
        },
        {
          'text' : mask_type,
          'color': 'mask',
        },
      ]
      processed_indices += [
        {
          'text' : sample_feed[i],
          'color': 'plain',
        },
        {
          'text' : mask_type,
          'color': 'mask',
        },
        {
          'text' : sample_indices[i].replace("\\n", "\n"),
          'color': 'prediction',
        },
      ]
    while i < len(sample_feed) - 1:
      i += 1
      processed_indices.append(
        {
          'text': sample_feed[i],
          'color': 'plain',
        },
      )
      processed_feed.append(
        {
          'text': sample_feed[i],
          'color': 'plain'
        }
      )
    sample.sample_indices = processed_indices
    sample.sample_feed = processed_feed

  sample_specs = {
    'summary'   : cached_models[target_sha]['summary'],
    'workspace' : workspace,
    'model_sha' : model_sha,
    'samples'   : sample_files,
  }
  return flask.render_template("sample_files.html", data = sample_specs, **GetBaseTemplateArgs())

@flask_app.route("/corpus/<int:corpus_id>/encoded/random/")
def random_encoded_contentfile(corpus_id: int):
  l.logger().debug("deeplearning.clgen.dashboard.random_encoded_contentfile()")
  (encoded_url,) = (
    db.session.query(dashboard_db.Corpus.encoded_url)
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )

  encoded_db = encoded.EncodedContentFiles(encoded_url, must_exist=True)

  with encoded_db.Session() as session:
    (random_id,) = (
      session.query(encoded.EncodedContentFile.id)
      .order_by(encoded_db.Random())
      .limit(1)
      .one()
    )

  return flask.redirect(f"/corpus/{corpus_id}/encoded/{random_id}/", code=302)

def Launch(host: str = "0.0.0.0",
           debug: bool = False,
           ):
  l.logger().debug("deeplearning.clgen.dashboard.Launch()")
  """Launch dashboard in a separate thread."""
  port = FLAGS.clgen_dashboard_port or portpicker.pick_unused_port()
  l.logger().info("Launching CLgen dashboard on http://{}:{}".format(host, port))
  kwargs = {
    "port": port,
    # Debugging must be disabled when run in a separate thread.
    "debug": debug,
    "host": host,
  }

  db.create_all()
  if debug:
    flask_app.run(**kwargs)
  else:
    thread = threading.Thread(
      target = flask_app.run, kwargs = kwargs
    )
    thread.setDaemon(True)
    thread.start()
    return thread
