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
import sqlalchemy as sql

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pbutil
from deeplearning.clgen import validation_database
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.dashboard import dashboard_db
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import internal_pb2
from absl import flags
import humanize

from eupy.native import logger as l

FLAGS = flags.FLAGS

# Disable flask banner on load.
_cli = sys.modules["flask.cli"]
_cli.show_server_banner = lambda *x: None

flags.DEFINE_integer(
  "clgen_dashboard_port", None, "The port to launch the server on.",
)

flask_app = flask.Flask(
  __name__,
  template_folder = environment.DASHBOARD_TEMPLATES,
  static_folder = environment.DASHBOARD_STATIC,
)

db = flask_sqlalchemy.SQLAlchemy(flask_app)

def GetBaseTemplateArgs():
  l.getLogger().debug("deeplearning.clgen.dashboard.GetBaseTemplateArgs()")
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

def parseCorpus(base_path):

  corpuses = []
  if (base_path / "corpus" / "encoded").exists():
    corpus_path = base_path / "corpus" / "encoded"
    for corpus_sha in corpus_path.iterdir():
      encoded_db = encoded.EncodedContentFiles("sqlite:///{}".format(corpus_sha / "encoded.db"), must_exist = True)
      corpuses.append(
        {
          'path': str(corpus_path / corpus_sha),
          'sha' : str(corpus_sha.stem),
          'datapoint_count': encoded_db.size,
          'summary': "{} datapoint corpus, {}".format(encoded_db.size, str(corpus_sha.stem))
        }
      )
  return corpuses

def parseModels(base_path):

  models = []
  if (base_path / "model").exists():
    for model_sha in (base_path / "model").iterdir():
      model_path = base_path / "model" / model_sha
      if (model_path / "META.pbtxt").exists():
        meta = parseMeta(model_path / "META.pbtxt")
        models.append(
          {          
            'path': str(model_path),
            'sha' : str(model_sha.stem),
            'config': meta,
            'training_log': parseTrainLogs(model_path / "logs"),
            'validation': parseValidationDB(model_path / "logs" / "validation_samples.db"),
            'samples': parseSamples(base_path, model_path / "samples"),
            'summary': parseModelSummary(meta)
          }
        )

  return models

def parseMeta(meta):
  with open(meta, 'r') as f:
    return f.read().splitlines()

def parseModelSummary(meta):
  m = pbutil.FromString('\n'.join(meta), internal_pb2.ModelMeta())
  if m.config.architecture.backend == model_pb2.NetworkArchitecture.TENSORFLOW_BERT:
    summary = ("TF_BERT, h_s: {}, num_h_l: {}, heads: {}, intm_s: {}, max_pemb: {}, max_preds: {}, dupe: {}, mask_prob: {}, target: {}"
        .format(m.config.architecture.hidden_size, m.config.architecture.num_hidden_layers, m.config.architecture.num_attention_heads,
        m.config.architecture.intermediate_size, m.config.architecture.max_position_embeddings, m.config.training.max_predictions_per_seq, 
        m.config.training.dupe_factor, round(m.config.training.masked_lm_prob, 3),
        "mask" if m.config.training.data_generator.HasField("mask") else "hole-{}".format(m.config.training.data_generator.hole.hole_length)
        )
      )
  else:
    summary = "TODO"
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

def parseSamples(base_path, sample_path):

  model_samplers = []

  if sample_path.exists():
    for sampler_sha in sample_path.iterdir():
      if (base_path / "sampler" / sampler_sha / "META.pbtxt").exists():
        for db_file in (sample_path / sampler_sha):
          if db_file.stem == "epoch_samples.db" or db_file.stem == "samples.db":
            samples_db = samples_database.SamplesDatabase("sqlite:///{}".format(db_file), must_exist = True)
            model_samplers.append({
                'sha': sampler_sha,
                'config': parseMeta(str(base_path / "sampler" / sampler_sha / "META.pbtxt")),
                'samples': ['todo', 'todo'],
              }
            )
  return model_samplers

data = {}
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
        'models': parseModels(p)
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
  # dummy_data = {
  #   "workspaces": {
  #     "corpuses": {
  #       'name': "haha", 
  #       'path': "xoxo", 
  #       'corpuses': ['1', '2', '3'], 
  #       'models': ['4', '5', '6']
  #       }
  #   },
  # }
  global data
  dummy_data = data
  return flask.render_template("corpus.html", data = dummy_data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/model_specs")
def model_specs(workspace: str, model_sha: str):
  global data
  if data == {}:
    data = parseData()
  current_model = {}

  for w in data['workspaces']:
    if data['workspaces'][w]['name'] == workspace:
      current_workspace = data['workspaces'][w]
      for mod in current_workspace['models']:
        if mod['sha'] == model_sha:
          current_model = mod
          break
  spec_data ={
    'config': current_model['config']
  }
  return flask.render_template("model_specs.html", data = spec_data, **GetBaseTemplateArgs())

@flask_app.route("/<string:workspace>/model/<string:model_sha>/validation")
def validation_samples(workspace: str, model_sha: str):
  global data
  if data == {}:
    data = parseData()
  current_model = {}

  for w in data['workspaces']:
    if data['workspaces'][w]['name'] == workspace:
      current_workspace = data['workspaces'][w]
      for mod in current_workspace['models']:
        if mod['sha'] == model_sha:
          current_model = mod
          break

  validation = current_model['validation']

  if validation['path']:
    # try:
    l.getLogger().error(validation['path'])
    val_db = validation_database.ValidationDatabase(str(validation['path']), must_exist = True)
    with val_db.Session() as session:
      validation['val_samples'] = session.query(validation_database.BERTValFile).all()
      # random.shuffle(validation['val_samples'])
    # except Exception as e:
    #   raise e

    for entry in validation['val_samples']:
      processed_input_ids = []
      if '[HOLE]' in entry.input_ids:
        mask_type = '[HOLE]'
      elif '[MASK]' in entry.input_ids:
        mask_type = '[MASK]'
      else:
        mask_type = ''
      input_ids = entry.input_ids.split(mask_type)
      mask_num = entry.num_targets
      # assert mask_num == len(input_ids) - 1, "{}, {}, {}".format(entry.input_ids, mask_num, len(input_ids))
      for i in range(mask_num):
        processed_input_ids += [
          {
            'text': input_ids[i],
            'color': 'plain',
            'length': len(input_ids[i]),
          },
          {
            'text': mask_type,
            'color': 'mask',
            'length': int(entry.masked_lm_lengths.split(',')[i]),
          },
          {
            'text': entry.masked_lm_predictions.split('\n')[i].replace(' ', '[ ]').replace('\n', '\\n'),
            'color': 'prediction',
            'length': 1,
          },
          {
            'text': entry.masked_lm_ids.split('\n')[i].replace(' ', '[ ]').replace('\n', '\\n'),
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
      # l.getLogger().info(entry.original_input)
      # l.getLogger().info(entry.input_ids)
      # l.getLogger().info(entry.masked_lm_predictions)
      # l.getLogger().info(entry.masked_lm_ids)
      # l.getLogger().warn(processed_input_ids)
      entry.input_ids = processed_input_ids
  validation['workspace'] = workspace
  validation['model_sha'] = model_sha
  return flask.render_template("validation_samples.html", data = validation, **GetBaseTemplateArgs())

@flask_app.route("/corpus/<int:corpus_id>/model/<int:model_id>/")
def report(corpus_id: int, model_id: int):
  corpus, corpus_config_proto, preprocessed_url, encoded_url = (
    db.session.query(
      dashboard_db.Corpus.summary,
      dashboard_db.Corpus.config_proto,
      dashboard_db.Corpus.preprocessed_url,
      dashboard_db.Corpus.encoded_url,
    )
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )
  model, model_config_proto = (
    db.session.query(
      dashboard_db.Model.summary, dashboard_db.Model.config_proto
    )
    .filter(dashboard_db.Model.id == model_id)
    .one()
  )

  telemetry = (
    db.session.query(
      dashboard_db.TrainingTelemetry.timestamp,
      dashboard_db.TrainingTelemetry.epoch,
      dashboard_db.TrainingTelemetry.step,
      dashboard_db.TrainingTelemetry.training_loss,
    )
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .all()
  )

  q1 = (
    db.session.query(sql.func.max(dashboard_db.TrainingTelemetry.id))
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .group_by(dashboard_db.TrainingTelemetry.epoch)
  )

  q2 = (
    db.session.query(
      dashboard_db.TrainingTelemetry.timestamp,
      dashboard_db.TrainingTelemetry.epoch,
      dashboard_db.TrainingTelemetry.step,
      dashboard_db.TrainingTelemetry.learning_rate,
      dashboard_db.TrainingTelemetry.training_loss,
      dashboard_db.TrainingTelemetry.pending,
    )
    .filter(dashboard_db.TrainingTelemetry.id.in_(q1))
    .order_by(dashboard_db.TrainingTelemetry.id)
  )

  q3 = (
    db.session.query(
      sql.sql.expression.cast(
        sql.func.avg(dashboard_db.TrainingTelemetry.ns_per_batch), sql.Integer
      ).label("us_per_step"),
    )
    .group_by(dashboard_db.TrainingTelemetry.epoch)
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .order_by(dashboard_db.TrainingTelemetry.id)
  )

  epoch_telemetry = [
    {
      "timestamp": r2.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
      "epoch": r2.epoch,
      "step": humanize.intcomma(r2.step),
      "learning_rate": f"{r2.learning_rate:.5E}",
      "training_loss": f"{r2.training_loss:.6f}",
      "pending": r2.pending,
      "us_per_step": humanize.naturaldelta(r3.us_per_step / 1e6),
    }
    for r2, r3 in zip(q2, q3)
  ]

  data = GetBaseTemplateArgs()
  data["corpus_id"] = corpus_id
  data["model_id"] = model_id
  data["corpus"] = corpus
  data["model"] = model
  data["data"] = {
    "corpus_config_proto": corpus_config_proto,
    "model_config_proto": model_config_proto,
    "telemetry": telemetry,
    "epoch_telemetry": epoch_telemetry,
    "preprocessed_url": preprocessed_url,
    "encoded_url": encoded_url,
  }
  data["urls"]["view_encoded_file"] = f"/corpus/{corpus_id}/encoded/random/"

  return flask.render_template("report.html", **data)


@flask_app.route("/corpus/<int:corpus_id>/encoded/random/")
def random_encoded_contentfile(corpus_id: int):
  l.getLogger().debug("deeplearning.clgen.dashboard.random_encoded_contentfile()")
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


@flask_app.route("/corpus/<int:corpus_id>/encoded/<int:encoded_id>/")
def encoded_contentfile(corpus_id: int, encoded_id: int):
  l.getLogger().debug("deeplearning.clgen.dashboard.encoded_contentfile()")
  (encoded_url,) = (
    db.session.query(dashboard_db.Corpus.encoded_url)
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )

  encoded_db = encoded.EncodedContentFiles(encoded_url, must_exist=True)

  with encoded_db.Session() as session:
    cf = (
      session.query(encoded.EncodedContentFile)
      .filter(encoded.EncodedContentFile.id == encoded_id)
      .limit(1)
      .one()
    )
    indices = cf.indices_array
    vocab = {
      v: k
      for k, v in encoded.EncodedContentFiles.GetVocabFromMetaTable(
        session
      ).items()
    }
    tokens = [vocab[i] for i in indices]
    text = "".join(tokens)
    encoded_cf = {
      "id": cf.id,
      "tokencount": humanize.intcomma(cf.tokencount),
      "indices": indices,
      "text": text,
      "tokens": tokens,
    }
    vocab = {
      "table": [(k, v) for k, v in vocab.items()],
      "size": len(vocab),
    }

  data = GetBaseTemplateArgs()
  data["encoded"] = encoded_cf
  data["vocab"] = vocab
  data["urls"]["view_encoded_file"] = f"/corpus/{corpus_id}/encoded/random/"
  return flask.render_template("encoded_contentfile.html", **data)


@flask_app.route(
  "/corpus/<int:corpus_id>/model/<int:model_id>/samples/<int:epoch>"
)
def samples(corpus_id: int, model_id: int, epoch: int):
  l.getLogger().debug("deeplearning.clgen.dashboard.samples()")
  samples = (
    db.session.query(
      dashboard_db.TrainingSample.sample,
      dashboard_db.TrainingSample.token_count,
      dashboard_db.TrainingSample.sample_time,
    )
    .filter(
      dashboard_db.TrainingSample.model_id == model_id,
      dashboard_db.TrainingSample.epoch == epoch,
    )
    .all()
  )

  data = {
    "samples": samples,
  }

  opts = GetBaseTemplateArgs()
  opts["urls"]["back"] = f"/corpus/{corpus_id}/model/{model_id}/"

  return flask.render_template(
    "samples.html", data=data, corpus_id=corpus_id, model_id=model_id, **opts
  )


def Launch(debug: bool = False):
  l.getLogger().debug("deeplearning.clgen.dashboard.Launch()")
  """Launch dashboard in a separate thread."""
  port = FLAGS.clgen_dashboard_port or portpicker.pick_unused_port()
  l.getLogger().info("Launching CLgen dashboard on http://127.0.0.1:{}".format(port))
  kwargs = {
    "port": port,
    # Debugging must be disabled when run in a separate thread.
    "debug": debug,
    "host": "0.0.0.0",
  }

  db.create_all()
  if debug:
    flask_app.run(**kwargs)
  else:
    thread = threading.Thread(target=flask_app.run, kwargs=kwargs)
    thread.setDaemon(True)
    thread.start()
    return thread
