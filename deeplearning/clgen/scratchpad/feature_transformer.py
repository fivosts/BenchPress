"""
Experimental transformer for feature space mapping.
"""
import math
import typing
import copy
import time
import pathlib
import typing
import tqdm
import multiprocessing
import pickle

from absl import app

from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.util import distributions
from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.models.torch_bert import optimizer
from deeplearning.benchpress.models.torch_bert import hooks
from deeplearning.benchpress.experiments import workers

torch = pytorch.torch

ENCODED_DB_PATH = "/home/foivos/unique_encoded.db"
TOKENIZER_PATH = "/home/foivos/backup_tokenizer.pkl"

class TransformerModel(torch.nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
               nlayers: int, pad_idx, dropout: float = 0.5):
    super().__init__()
    self.model_type  = 'Transformer'
    self.embed       = torch.nn.Embedding(ntoken, d_model, padding_idx = pad_idx)
    self.pos_encoder = PositionalEncoding(d_model, dropout)

    self.target_embed       = torch.nn.Embedding(ntoken, d_model)
    self.target_pos_encoder = PositionalEncoding(d_model, dropout)

    self.d_model     = d_model

    encoder_layers           = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first = True)
    encoder_norm             = torch.nn.LayerNorm(d_model, eps=1e-5)
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, encoder_norm)

    decoder_layer            = torch.nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first = True)
    decoder_norm             = torch.nn.LayerNorm(d_model, eps=1e-5)
    self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
    self.linear = torch.nn.Linear(d_model, ntoken)

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.embed.weight.data.uniform_(-initrange, initrange)
    # self.decoder.bias.data.zero_()
    # self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: torch.Tensor, target: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask = None) -> torch.Tensor:
    """
    Args:
        src: Tensor, shape [seq_len, batch_size]
        src_mask: Tensor, shape [seq_len, seq_len]

    Returns:
        output Tensor of shape [seq_len, batch_size, ntoken]
    """
    src1 = self.embed(src) * math.sqrt(self.d_model)
    src2 = self.pos_encoder(src1)
    output1 = self.transformer_encoder(src2, mask = src_mask, src_key_padding_mask = src_key_padding_mask)
    tgt1 = self.embed(target) * math.sqrt(self.d_model)
    tgt2 = self.pos_encoder(tgt1)

    output2 = self.transformer_decoder(tgt2, output1)
    output3 = self.linear(output2)
    # print(src.shape)
    # print(src1.shape)
    # print(src2.shape)
    # print(output1.shape)
    # print(output2.shape)
    # print(output3.shape)
    # input()
    return output3

class PositionalEncoding(torch.nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = torch.nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class FeatureDataset(torch.utils.data.Dataset):
  def __init__(self, corpus: typing.List[typing.Dict[str, typing.Dict[str, float]]]) -> None:
    self.corpus = corpus
    self.feat_tokenizer = tokenizers.FeatureTokenizer.FromFeatures(768, 65536, 2048)
    self.dataset = self.compute_dataset()
    return

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx: int):
    if idx < 0:
      if -idx > len(self):
        raise ValueError
      idx = len(self) + idx
    return self.dataset[idx]

  def compute_dataset(self):
    seq_len   = 256

    f_len = {
      "GreweFeatures": 6,
      "AutophaseFeatures": 56,
      "InstCountFeatures": 70,
    }
    pad_len = seq_len - sum(list(f_len.values()))
    dataset   = []

    for dp in self.corpus:
      for fspace in {"GreweFeatures", "AutophaseFeatures", "InstCountFeatures"}:

        inp = []
        for n, x in dp[fspace].items():
          if n not in {"F2:coalesced/mem", "F4:comp/mem"}:
            try:
              x = int(x)
            except Exception:
              continue
            inp.append(self.feat_tokenizer.TokenizeFeature(int(x)))
        assert len(inp) == f_len[fspace], len(inp)

        target_feats = dp["AutophaseFeatures"]
        target = []
        for x in target_feats.values():
          try:
            x = int(x)
          except Exception:
            continue
          target.append(self.feat_tokenizer.TokenizeFeature(int(x)))
        assert len(target) == f_len["AutophaseFeatures"], len(target)

        if fspace == "GreweFeatures":
          d = {
            'inputs'         : torch.LongTensor(inp + [self.feat_tokenizer.padToken] * (f_len["AutophaseFeatures"] + f_len["InstCountFeatures"] + pad_len)),
            'target'         : torch.LongTensor(target)
          }
        elif fspace == "AutophaseFeatures":
          d = {
            'inputs'        : torch.LongTensor([self.feat_tokenizer.padToken] * f_len["GreweFeatures"] + inp + [self.feat_tokenizer.padToken] * (f_len["InstCountFeatures"] + pad_len)),
            'target'        : torch.LongTensor(target)
          }
        else:
          d = {
            'inputs'         : torch.LongTensor([self.feat_tokenizer.padToken] * (f_len["GreweFeatures"] + f_len["AutophaseFeatures"]) + inp + [self.feat_tokenizer.padToken] * pad_len),
            'target'         : torch.LongTensor(target)
          }
        d['padding_mask'] = d['inputs'] == self.feat_tokenizer.padToken
        dataset.append(d)
    return dataset

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
  """Generates an upper-triangular matrix of -inf, with zeros on diag."""
  return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def data_process(raw_text_iter: torch.utils.data.dataset.IterableDataset) -> torch.Tensor:
  """Converts raw text into a flat Tensor."""
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
  """Divides the data into bsz separate sequences, removing extra elements
  that wouldn't cleanly fit.

  Args:
      data: Tensor, shape [N]
      bsz: int, batch size

  Returns:
      Tensor of shape [N // bsz, bsz]
  """
  seq_len = data.size(0) // bsz
  data = data[:seq_len * bsz]
  data = data.view(bsz, seq_len).t().contiguous()
  return data.to(device)

def get_data_features(db, tokenizer, size_limit = None) -> typing.List[typing.Dict[str, typing.Dict[str, float]]]:
  """
  Get or set feature with data list of tuples.
  """
  datapoints = []
  db_feats = db.get_data_features(tokenizer, size_limit)
  for inp in tqdm.tqdm(db_feats, total = len(db_feats), desc = "Fetch data"):
    feats = workers.ContentFeat(inp)
    if len(inp) == 2:
      src, _ = inp
      include = ""
    else:
      src, include, _ = inp
    try:
      datapoints.append({
        "GreweFeatures"     : feats["GreweFeatures"],
        "AutophaseFeatures" : feats["AutophaseFeatures"],
        "InstCountFeatures" : feats["InstCountFeatures"],
      })
    except KeyError as e:
      l.logger().warn(e)
  return datapoints

def Train(feat_vecs):
  size = len(feat_vecs)
  train_data, val_data = feat_vecs[:(9 * size) // 10], feat_vecs[(9 * size) // 10:]
  device = 'cuda'

  num_epochs = 30
  batch_size = 32
  num_warmup_steps = 5000
  learning_rate    = 45 / 1e6

  train_dataset = FeatureDataset(train_data)
  val_dataset   = FeatureDataset(val_data)


  vocab_size = len(train_dataset.feat_tokenizer)
  emsize  = 64  # embedding dimension
  d_hid   = 128  # dimension of the feedforward network model in nn.TransformerEncoder
  nlayers = 2   # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead   = 2   # number of heads in nn.MultiheadAttention
  dropout = 0.1 # dropout probability
  model = TransformerModel(
    vocab_size,
    emsize,
    nhead,
    d_hid,
    nlayers,
    train_dataset.feat_tokenizer.padToken,
    dropout
  ).to(device)

  ## Define dataloaders.
  train_loader = torch.utils.data.dataloader.DataLoader(
    dataset     = train_dataset,
    batch_size  = batch_size,
    sampler     = torch.utils.data.RandomSampler(train_dataset, replacement = False),
    num_workers = 0,
    drop_last   = False,
  )
  val_loader = torch.utils.data.dataloader.DataLoader(
    dataset     = val_dataset,
    batch_size  = batch_size,
    sampler     = torch.utils.data.RandomSampler(val_dataset, replacement = False),
    num_workers = 0,
    drop_last   = False,
  )
  ## Also create scheduler and optmizer.
  opt, scheduler = optimizer.create_optimizer_and_scheduler(
    model           = model,
    num_train_steps = (num_epochs * len(train_dataset)) // batch_size,
    warmup_steps    = num_warmup_steps,
    learning_rate   = learning_rate,
  )
  loss_fn = torch.nn.CrossEntropyLoss()
  model.zero_grad()

  hook_path = pathlib.Path("./feat_reconstruction").resolve()
  hook_path.mkdir(exist_ok = True, parents = True)
  train_hook = hooks.tensorMonitorHook(hook_path, 0, 50)
  val_hook = hooks.tensorMonitorHook(pathlib.Path("./feat_reconstruction").resolve(), 0, 10)

  for ep in tqdm.tqdm(range(num_epochs), desc = "Epoch", leave = False):
    model.train()
    for batch in tqdm.tqdm(train_loader, total = len(train_loader), desc = "Batch", leave = False):
      inp, att, target = batch['inputs'], batch['padding_mask'], batch['target']

      output = model(inp.to(device), target.to(device), src_key_padding_mask = att.to(device))
      loss = loss_fn(output.view(-1, len(train_dataset.feat_tokenizer)), target.to(device).view(-1))

      opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      opt.step()
      scheduler.step()

      train_hook.step(total_loss = loss.item())
    l.logger().info("Epoch {} loss {}".format(ep, train_hook.epoch_loss))
    train_hook.end_epoch()

    model.eval()

    for batch in tqdm.tqdm(train_loader, total = len(train_loader), desc = "Val Train Batch", leave = False):
      inp, att, target = batch['inputs'], batch['padding_mask'], batch['target']

      output = model(inp.to(device), target.to(device), src_key_padding_mask = att.to(device))
      loss = loss_fn(output.view(-1, len(train_dataset.feat_tokenizer)), target.to(device).view(-1))

      euclids = []
      accuracy = []
      for bi in range(output.size(0)):
        raw_out = torch.argmax(output[bi], dim = 1).cpu()
        targ    = target[bi].cpu()
        assert len(raw_out) == len(targ), "{}/{}".format(len(raw_out), len(targ))
        dist = 0.0
        for vi in range(len(targ)):
          dist += (targ[vi] - raw_out[vi])**2
        euclids.append(math.sqrt(dist))
        accuracy.append(len(torch.where(targ == raw_out)[0]) / len(targ))
      mean_dist = sum(euclids) / len(euclids)
      mean_accuracy = sum(accuracy) / len(accuracy)
      val_hook.step(val_train_loss = loss.item(), val_train_dist = mean_dist, val_train_accuracy = mean_accuracy)

    for batch in tqdm.tqdm(val_loader, total = len(val_loader), desc = "Val Batch", leave = False):
      inp, att, target = batch['inputs'], batch['padding_mask'], batch['target']

      output = model(inp.to(device), target.to(device) ) #, src_key_padding_mask = att.to(device))
      loss = loss_fn(output.view(-1, len(train_dataset.feat_tokenizer)), target.to(device).view(-1))

      euclids = []
      accuracy = []
      for bi in range(output.size(0)):
        raw_out = torch.argmax(output[bi], dim = 1).cpu()
        targ    = target[bi].cpu()
        assert len(raw_out) == len(targ), "{}/{}".format(len(raw_out), len(targ))
        dist = 0.0
        for vi in range(len(targ)):
          dist += (targ[vi] - raw_out[vi])**2
        euclids.append(math.sqrt(dist))
        accuracy.append(len(torch.where(targ == raw_out)[0]) / len(targ))

      mean_dist = sum(euclids) / len(euclids)
      mean_accuracy = sum(accuracy) / len(accuracy)
      val_hook.step(val_loss = loss.item(), val_dist = mean_dist, val_accuracy = mean_accuracy)
  return

def Validate(model, tokenizer, train_loader, val_loader):


  return

def main(*args):

  db = encoded.EncodedContentFiles(url = "sqlite:///{}".format(ENCODED_DB_PATH), must_exist = True)
  tokenizer = tokenizers.TokenizerBase.FromFile(pathlib.Path(TOKENIZER_PATH).resolve())
  feat_vecs = get_data_features(db, tokenizer)

  Train(feat_vecs)
  return

if __name__ == "__main__":
  app.run(main)  
