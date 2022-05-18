import typing

from deeplearning.clgen.util import pytorch
torch  = pytorch.torch

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def make_sentinel(i):
  # signals (1) a location to insert an infill and (2) the start of the infill generation
  return f"<|mask:{i}|>"

def generate(model: torch.nn.Module, inp: str, tokenizer, max_to_generate: int=128, temperature: float=0.2):
  """
  Do standard left-to-right completion of the prefix `input` by sampling from the model
  """
  input_ids = tokenizer(inp, return_tensors="pt").input_ids.to(pytorch.device)
  max_length = max_to_generate + input_ids.flatten().size(0)
  if max_length > 2048:
    print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
  with torch.no_grad():
    output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
  detok_hypo_str = tokenizer.decode(output.flatten())
  if detok_hypo_str.startswith(BOS):
    detok_hypo_str = detok_hypo_str[len(BOS):]
  return detok_hypo_str

def infill(model, inp: str, tokenizer, max_to_generate: int=128, temperature: float=0.7, extra_sentinel: bool=True, max_retries: int=1):
  """
  Generate infills to complete a partial document, e.g.
  [A C E] -> [A B C D E], where B and D are infills that have been generated.

  parts: str. One string instance to input for sampling.
  max_to_generate: int. maximum number of tokens to generate. Keep in mind
          that the model context size is 2048.
  temperature: float. temperature parameter for sampling.
  extra_sentinel: bool. we recommend setting this to True, as it makes it
          easier for the model to end generated infills. See the footnote in 
          section 2.2 of our paper for details.
  max_retries: int. if > 1, use rejection sampling to keep sampling infills until
          all infills sample a completion token.

  returns a dictionary containing the following:
      text:  str, the completed document (with infills inserted)
      parts:  List[str], length N. Same as passed to the method
      infills:  List[str], length N-1. The list of infills generated
      retries_attempted:  number of retries used (if max_retries > 1)
  """
  parts = inp.split('<insert>')
  assert isinstance(parts, list)
  retries_attempted = 0
  done = False

  while (not done) and (retries_attempted < max_retries):
    retries_attempted += 1
    infills = []
    complete = []
  
    ## (1) build the prompt
    if len(parts) == 1:
      prompt = parts[0]
      completion = generate(model, prompt, tokenizer, max_to_generate, temperature)
      # completion = completion[len(prompt):]
      if EOM not in completion:
        completion += EOM
      completion = completion[:completion.index(EOM) + len(EOM)]
      infilled = completion[:-len(EOM)]
      infills.append(infilled)
      return {
          'text': completion, # str, the completed document (with infills inserted)
          'parts': parts, # List[str], length N. Same as passed to the method
          'infills': infills, # List[str], length N-1. The list of infills generated
          'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
      }
    else:
      prompt = ""
      # encode parts separated by sentinel
      for sentinel_ix, part in enumerate(parts):
          prompt += part
          if extra_sentinel or (sentinel_ix < len(parts) - 1):
              prompt += make_sentinel(sentinel_ix)
    done = True
    ## (2) generate infills
    for sentinel_ix, part in enumerate(parts[:-1]):
      complete.append(part)
      prompt += make_sentinel(sentinel_ix)
      # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
      completion = generate(model, prompt, tokenizer, max_to_generate, temperature)
      completion = completion[len(prompt):]
      if EOM not in completion:
        completion += EOM
        done = False
      completion = completion[:completion.index(EOM) + len(EOM)]
      infilled = completion[:-len(EOM)]
      infills.append(infilled)
      complete.append(infilled)
      prompt += completion
    complete.append(parts[-1])
    text = ''.join(complete)

  return {
    'text': text, # str, the completed document (with infills inserted)
    'parts': parts, # List[str], length N. Same as passed to the method
    'infills': infills, # List[str], length N-1. The list of infills generated
    'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
  } 
