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
import torchtext

from absl import app

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import distributions
from deeplearning.clgen.experiments import workers

torch = pytorch.torch

class TransformerModel(torch.nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
               nlayers: int, pad_idx, dropout: float = 0.5):
    super().__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
    self.encoder = torch.nn.Embedding(ntoken, d_model)
    self.d_model = d_model
    self.decoder = torch.nn.Linear(d_model, ntoken)

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_idx = None) -> torch.Tensor:
    """
    Args:
        src: Tensor, shape [seq_len, batch_size]
        src_mask: Tensor, shape [seq_len, seq_len]

    Returns:
        output Tensor of shape [seq_len, batch_size, ntoken]
    """
    src = self.encoder(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, mask = src_mask, src_key_padding_idx = src_key_padding_idx)
    output = self.decoder(output)
    return output

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
    return self.dataset[sidx]

  def compute_dataset(self):
    seq_len   = 256

    f_len = {
      "GreweFeatures": 6,
      "AutophaseFeatures": 52,
      "InstCountFeatures": 71,
    }
    pad_len = seq_len - sum(list(f_len.values()))
    dataset   = []

    for dp in self.corpus:
      for fspace in {"GreweFeatures", "AutophaseFeatures", "InstCountFeatures"}:

        inp = []
        for x in dp[fspace].values():
          try:
            x = int(x)
          except Exception:
            continue
          inp.append(self.tokenizeFeature(int(x)))
        assert len(inp) == f_len[fspace]

        target_feats = dp["AutophaseFeatures"]
        target = []
        for x in target_feats.values():
          try:
            x = int(x)
          except Exception:
            continue
          target.append(self.tokenizeFeature(int(x)))
        assert len(target) == f_len["AutophaseFeatures"]

        if fspace == "GreweFeatures":
          d = {
            'inputs'         : torch.LongTensor(inp + [self.tokenizer.padToken] * (f_len["AutophaseFeatures"] + f_len["InstCountFeatures"] + pad_len)),
            'target'         : torch.LongTensor(target)
          }
        elif fspace == "AutophaseFeatures":
          d = {
            'inputs'        : torch.LongTensor([self.tokenizer.padToken] * f_len["GreweFeatures"] + inp + [self.tokenizer.padToken] * (f_len["InstCountFeatures"] + pad_len)),
            'target'        : torch.LongTensor(target)
          }
        else:
          d = {
            'inputs'         : torch.LongTensor([self.tokenizer.padToken] * (f_len["GreweFeatures"] + f_len["AutophaseFeatures"]) + inp + [self.tokenizer.padToken] * pad_len),
            'target'         : torch.LongTensor(target)
          }
        d['padding_mask'] = d['inputs'] != self.tokenizer.padToken
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
    datapoints.append({
      "GreweFeatures"     : feats["GreweFeatures"],
      "AutophaseFeatures" : feats["AutophaseFeatures"],
      "InstCountFeatures" : feats["InstCountFeatures"],
    })
  return datapoints

def main(*args):

  db = encoded.EncodedContentFiles(url = "sqlite:///{}".format(ENCODED_DB_PATH), must_exist = True)
  tokenizer = tokenizers.TokenizerBase.FromFile(pathlib.Path(TOKENIZER_PATH).resolve())
  feat_vecs = get_data_features(db, tokenizer)

  data = process_data(feat_vecs)
  size = len(data)
  train_data, val_data = data[:(9 * size) // 10], data[(9 * size) // 10:]

  num_epochs = 30
  batch_size = 32
  num_warmup_steps = 5000
  learning_rate    = 45 / 1e6

  train_dataset = FeatureDataset(train_data)
  val_dataset   = FeatureDataset(val_data)


  vocab_size = len(train_dataset.feat_tokenizer)
  emsize  = 64  # embedding dimension
  d_hid   = 256  # dimension of the feedforward network model in nn.TransformerEncoder
  nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead   = 2  # number of heads in nn.MultiheadAttention
  dropout = 0.1  # dropout probability
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

  for ep in tqdm.tqdm(num_epochs, desc = "Epoch", leave = True):
    model.train()
    for batch in tqdm.tqdm(train_loader, total = len(train_loader), desc = "Batch", leave = True):
      inp, att, target = batch['inputs'], batch['padding_mask'], batch['target']

      ## Insert model step here
      output = model(data, src_key_padding_idx = att)

      ## Insert target cross entropy calculation.

      loss = loss_fn(output.view(-1, len(train_dataset.feat_tokenizer)), targets)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optimizer.step()
      scheduler.step()

      loss_val = loss.item()
      print(loss_val)
  return

if __name__ == "__main__":
  app.run(main)  



# train_iter = torchtext.datasets.WikiText2(split='train')
# tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
# vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
# vocab.set_default_index(vocab['<unk>'])

# # train_iter was "consumed" by the process of building the vocab,
# # so we have to create it again
# train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
# train_data = data_process(train_iter)
# val_data = data_process(val_iter)
# test_data = data_process(test_iter)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch_size = 10
# eval_batch_size = 10
# train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

# bptt = 35

# ntokens = len(vocab)  # size of vocabulary
# emsize = 200  # embedding dimension
# d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2  # number of heads in nn.MultiheadAttention
# dropout = 0.2  # dropout probability
# model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


# criterion = torch.nn.CrossEntropyLoss()
# lr = 5.0  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# best_val_loss = float('inf')
# epochs = 3
# best_model = None


# def get_batch(source: torch.Tensor, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
#   """
#   Args:
#       source: Tensor, shape [full_seq_len, batch_size]
#       i: int

#   Returns:
#       tuple (data, target), where data has shape [seq_len, batch_size] and
#       target has shape [seq_len * batch_size]
#   """
#   print()
#   print()
#   print()
#   print()
#   print(source.shape)
#   seq_len = min(bptt, len(source) - 1 - i)
#   print(seq_len)
#   data = source[i:i+seq_len]
#   print(data.shape)
#   target = source[i+1:i+1+seq_len].reshape(-1)
#   print(target.shape)
#   print(data)
#   print(target)
#   for b in data:
#     print(vocab.lookup_tokens([int(x) for x in b]))
#   print(vocab.lookup_tokens([int(x) for x in target]))
#   input()
#   return data, target

# def train(model: torch.nn.Module) -> None:
#   model.train()  # turn on train mode
#   total_loss = 0.
#   log_interval = 200
#   start_time = time.time()
#   src_mask = generate_square_subsequent_mask(bptt).to(device)

#   num_batches = len(train_data) // bptt
#   for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
#     print()
#     print(batch)
#     print(i)
#     data, targets = get_batch(train_data, i)
#     batch_size = data.size(0)
#     if batch_size != bptt:  # only on last batch
#         src_mask = src_mask[:batch_size, :batch_size]
#     output = model(data, src_mask)
#     print(output.view(-1, ntokens).shape)
#     input()
#     loss = criterion(output.view(-1, ntokens), targets)

#     optimizer.zero_grad()
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()

#     total_loss += loss.item()
#     if batch % log_interval == 0 and batch > 0:
#       lr = scheduler.get_last_lr()[0]
#       ms_per_batch = (time.time() - start_time) * 1000 / log_interval
#       cur_loss = total_loss / log_interval
#       ppl = math.exp(cur_loss)
#       print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
#             f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
#             f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
#       total_loss = 0
#       start_time = time.time()

# def evaluate(model: torch.nn.Module, eval_data: torch.Tensor) -> float:
#   model.eval()  # turn on evaluation mode
#   total_loss = 0.
#   src_mask = generate_square_subsequent_mask(bptt).to(device)
#   with torch.no_grad():
#     for i in range(0, eval_data.size(0) - 1, bptt):
#       data, targets = get_batch(eval_data, i)
#       batch_size = data.size(0)
#       if batch_size != bptt:
#         src_mask = src_mask[:batch_size, :batch_size]
#       output = model(data, src_mask)
#       output_flat = output.view(-1, ntokens)
#       total_loss += batch_size * criterion(output_flat, targets).item()
#   return total_loss / (len(eval_data) - 1)

# for epoch in range(1, epochs + 1):
#   epoch_start_time = time.time()
#   train(model)
#   val_loss = evaluate(model, val_data)
#   val_ppl = math.exp(val_loss)
#   elapsed = time.time() - epoch_start_time
#   print('-' * 89)
#   print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
#         f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
#   print('-' * 89)

#   if val_loss < best_val_loss:
#     best_val_loss = val_loss
#     best_model = copy.deepcopy(model)

#   scheduler.step()
