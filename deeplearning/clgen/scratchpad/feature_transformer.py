"""
Experimental transformer for feature space mapping.
"""
import math
import typing
import copy
import time

import torchtext
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

train_iter = torchtext.datasets.WikiText2(split='train')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

class TransformerModel(torch.nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
               nlayers: int, dropout: float = 0.5):
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

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        src: Tensor, shape [seq_len, batch_size]
        src_mask: Tensor, shape [seq_len, seq_len]

    Returns:
        output Tensor of shape [seq_len, batch_size, ntoken]
    """
    src = self.encoder(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, src_mask)
    output = self.decoder(output)
    print(output.shape)
    print(len(vocab))
    input()
    return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
  """Generates an upper-triangular matrix of -inf, with zeros on diag."""
  return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

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
  print(data.shape)
  seq_len = data.size(0) // bsz
  print(data.size(0))
  print(seq_len)
  data = data[:seq_len * bsz]
  print(data.shape)
  data = data.view(bsz, seq_len).t().contiguous()
  print(data.shape)
  return data.to(device)


# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 35

ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


criterion = torch.nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float('inf')
epochs = 3
best_model = None


def get_batch(source: torch.Tensor, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
  """
  Args:
      source: Tensor, shape [full_seq_len, batch_size]
      i: int

  Returns:
      tuple (data, target), where data has shape [seq_len, batch_size] and
      target has shape [seq_len * batch_size]
  """
  print()
  print()
  print()
  print()
  print(source.shape)
  seq_len = min(bptt, len(source) - 1 - i)
  print(seq_len)
  data = source[i:i+seq_len]
  print(data.shape)
  target = source[i+1:i+1+seq_len].reshape(-1)
  print(target.shape)
  print(data)
  print(target)
  for b in data:
    print(vocab.lookup_tokens([int(x) for x in b]))
  print(vocab.lookup_tokens([int(x) for x in target]))
  input()
  return data, target

def train(model: torch.nn.Module) -> None:
  model.train()  # turn on train mode
  total_loss = 0.
  log_interval = 200
  start_time = time.time()
  src_mask = generate_square_subsequent_mask(bptt).to(device)

  num_batches = len(train_data) // bptt
  for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    print()
    print(batch)
    print(i)
    data, targets = get_batch(train_data, i)
    batch_size = data.size(0)
    if batch_size != bptt:  # only on last batch
        src_mask = src_mask[:batch_size, :batch_size]
    output = model(data, src_mask)
    print(output.view(-1, ntokens).shape)
    input()
    loss = criterion(output.view(-1, ntokens), targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    if batch % log_interval == 0 and batch > 0:
      lr = scheduler.get_last_lr()[0]
      ms_per_batch = (time.time() - start_time) * 1000 / log_interval
      cur_loss = total_loss / log_interval
      ppl = math.exp(cur_loss)
      print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
      total_loss = 0
      start_time = time.time()

def evaluate(model: torch.nn.Module, eval_data: torch.Tensor) -> float:
  model.eval()  # turn on evaluation mode
  total_loss = 0.
  src_mask = generate_square_subsequent_mask(bptt).to(device)
  with torch.no_grad():
    for i in range(0, eval_data.size(0) - 1, bptt):
      data, targets = get_batch(eval_data, i)
      batch_size = data.size(0)
      if batch_size != bptt:
        src_mask = src_mask[:batch_size, :batch_size]
      output = model(data, src_mask)
      output_flat = output.view(-1, ntokens)
      total_loss += batch_size * criterion(output_flat, targets).item()
  return total_loss / (len(eval_data) - 1)

for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train(model)
  val_loss = evaluate(model, val_data)
  val_ppl = math.exp(val_loss)
  elapsed = time.time() - epoch_start_time
  print('-' * 89)
  print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
  print('-' * 89)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = copy.deepcopy(model)

  scheduler.step()
