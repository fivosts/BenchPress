# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file contains the definition of atomizers.

An atomizer converts a block of text into a sequence of vocbulary tokens.
"""
import pathlib
import pickle
import typing
from collections import Counter

import numpy as np


from absl import flags
from labm8.py import labdate

from eupy.native import logger as l

FLAGS = flags.FLAGS

def FromText(config: corpus_pb2.Corpus.atomizer, corpus_txt: str):
  mask_atoms = False if config.mask_atoms is None else config.mask_atoms

  if config.token_types   == "character":
    if config.token_list is not None:
      l.getLogger().warning("token list in character-based tokenization is going to be ignored.")
    return AsciiCharacterAtomizer.FromText(corpus_txt, mask_atoms)
  elif config.token_types == "word":
    wpc_tok = False if config.wordpiece_tokenization is None else config.wordpiece_tokenization
    return WordAtomizer.FromText(corpus_text, 
                                 config.token_list, 
                                 mask_atoms, 
                                 wpc_tok
                                )
  else:
    raise NotImplementedError

class AtomizerBase(object):
  """The base class for implementing atomizers."""

  def __init__(self, 
               vocab: typing.Dict[str, int],
               metaTokens: typing.Dict[str, str]
               ):
    """Instantiate an atomizer.

    Args:
      vocab: A dictionary of mappings from character sequences (atoms) into
        indices.
      metaTokens: A dictionary mapping the metaTokens needed for tokenization.
        (Used when masking is selected)
    Raises:
      TypeError: If vocab is not a dictionary.
      ValueError: If the dictionary of mappings includes any duplicate values.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.__init__()")
    self.vocab      = vocab
    self.metaTokens = metaTokens
    self._UpdateVocabulary()

  @property
  def atoms(self) -> typing.List[str]:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.atoms()")
    """A list of atoms in the vocabulary."""
    return list(sorted(self.vocab.keys()))

  @property
  def indices(self) -> typing.List[int]:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.indices()")
    """A list of vocabulary indices."""
    return list(sorted(self.vocab.values()))

  def _UpdateVocabulary(self) -> None:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase._UpdateVocabulary()")
    """Private method which must be called if vocab is modified."""
    if not isinstance(self.vocab, dict):
      raise TypeError("vocabulary must be a dict")

    # Each atom and index must be unique to ensure deterministic encoding.
    if len(set(self.vocab.keys())) != len(self.vocab):
      raise ValueError("all atoms must be unique")
    if len(set(self.vocab.values())) != len(self.vocab):
      raise ValueError("all indices must be unique")

    self.vocab_size = len(self.vocab)
    self.decoder = {val: key for key, val in self.vocab.items()}

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.

    Raises:
      VocabError: If the input text contains elements not in the vocabulary.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.AtomizeString()")
    raise NotImplementedError("abstract class")

  def TokenizeString(self, text: str) -> typing.List[str]:
    """Split the text into atoms, but do not encode to indices.

    Args:
      text: Input text.

    Returns:
      A list of tokens.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.TokenizeString()")
    indices = self.AtomizeString(text)
    return list(map(lambda x: self.decoder[x], indices))

  def DeatomizeIndices(self, encoded: np.array):
    """Translate atomized code back into a string.

    Args:
      encoded: An nparray of encoded vocabulary indices.

    Returns:
      The decoded text.
      Returns string if nparray is one-dimensional.
      Else returns list for each extra dimension of strings.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.DeatomizeIndices()")
    try:
      if np.ndim(encoded) > 1:
        return [ self.DeatomizeIndices(x) for x in encoded ]
      elif np.ndim(encoded) == 1:
        return "".join(list(map(lambda x: self.decoder[x], encoded)))
      else:
        raise ValueError("Wrong encoded array specified")
    except KeyError:
      raise KeyError("Out of vocab: {}".format(encoded))

  def ToFile(self, path: pathlib.Path) -> None:
    """Save an atomizer to file."""
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.ToFile()")
    with open(path, "wb") as f:
      pickle.dump(self, f)

  @classmethod
  def FromText(cls, text: str) -> "AtomizerBase":
    """Instantiate and specialize an atomizer from a corpus text.

    Args:
      text: Text corpus

    Returns:
      An atomizer instance.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.FromText()")
    raise NotImplementedError("abstract class")

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> "AtomizerBase":
    """Load an atomizer from file."""
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.FromFile()")
    with open(path, "rb") as infile:
      return pickle.load(infile)


class AsciiCharacterAtomizer(AtomizerBase):
  """An atomizer for character-level syntactic modelling."""

  def AtomizeString(self, text: str, metaTokens: typing.Dict[str, str]) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.AtomizeString()")
    try:
      if not self.metaTokens:
        return np.array(list(map(lambda x: self.vocab[x], text)), dtype=np.int32)
      else:
        encoded = []
        skipNext = 0
        for idx, char in enumerate(text):
          if skipNext > 0:
            skipNext -= 1
            continue
          if char == '[':
            for meta in self.metaTokens.values():
              if text[idx: idx + len(meta)] == meta:
                encoded.append(self.vocab[meta])
                skipNext = len(meta) - 1
                break
          if skipNext == 0:
            encoded.append(self.vocab[char])
        return np.array(encoded, dtype = np.int32)
    except KeyError:
      raise ValueError("OoV index in string tokenizing.")
      
  def __repr__(self) -> str:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.__repr__()")
    return f"AsciiCharacterAtomizer[{self.vocab_size} chars]"

  @classmethod
  def FromText(cls, text: str, mask_atoms: bool = False) -> "AsciiCharacterAtomizer":
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus.

    Returns:
      An atomizer instance.
    """
    if mask_atoms:
      metaTokens = {
        '[START]'   : '[START]',
        '[END]'     : '[END]',
        '[PAD]'     : '[PAD]',
        '[MASK]'    : '[MASK]',
        '[HOLE]'    : '[HOLE]',
        '[ENDHOLE]' : '[ENDHOLE]',
      }
    else:
      metaTokens = {}
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.FromText()")
    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    atoms, _ = zip(*count_pairs)
    atoms = tuple(metaTokens.keys()) + atoms
    vocab = dict(zip(atoms, range(len(atoms))))
    return AsciiCharacterAtomizer(vocab, metaTokens)

class WordAtomizer(AtomizerBase):
  """A greedy atomizer supports multi-character tokens."""

  def __init__(self, vocab: typing.Dict[str, int], determine_chars=False):
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.WordAtomizer.__init__()")
    self.determine_chars = determine_chars
    super(WordAtomizer, self).__init__(vocab)

    multichars = set(k for k in self.atoms if len(k) > 1)
    first_chars = set(a[0] for a in multichars)
    self.lookup = dict(
      (c, [a for a in multichars if a[0] == c]) for c in first_chars
    )

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.WordAtomizer.AtomizeString()")

    def _AddToVocab(token: str) -> int:
      """Add a token to the vocabulary and return its index."""
      l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.WordAtomizer._AddToVocab()")
      if self.determine_chars and token not in self.vocab:
        max_index = max(self.vocab.values())
        self.vocab[token] = max_index + 1
      return self.vocab[token]

    indices = []
    i = 0
    j = 2
    try:
      while i < len(text):
        if self.lookup.get(text[i]):
          if j <= len(text) and any(
            x.startswith(text[i:j]) for x in self.lookup[text[i]]
          ):
            j += 1
          else:
            while j > i + 1:
              if any(x == text[i:j] for x in self.lookup[text[i]]):
                indices.append(self.vocab[text[i:j]])
                i = j
                j += 2
                break
              else:
                j -= 1
            else:
              indices.append(_AddToVocab(text[i]))
              i += 1
              j += 2
        else:
          indices.append(_AddToVocab(text[i]))
          i += 1
          j += 2
    except KeyError:
      raise ValueError

    if self.determine_chars:
      self._UpdateVocabulary()

    return np.array(indices, dtype=np.int32)

  def __repr__(self) -> str:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.WordAtomizer.__repr__()")
    return f"WordAtomizer[{self.vocab_size} tokens]"

  @classmethod
  def FromText(cls, text: str, atoms: typing.Set[str]) -> "WordAtomizer":
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus
      atoms: A list of multi-character tokens.

    Returns:
      An atomizer instance.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.WordAtomizer.FromText()")
    if not atoms:
      raise ValueError("No atoms specified")

    # Instantiate a greedy atomizer using the full vocabulary.
    full_vocab = dict(zip(atoms, range(len(atoms))))
    c = WordAtomizer(full_vocab, determine_chars=True)
    # Derive the subset of the vocabulary required to encode the given text.
    tokens = sorted(list(set(c.TokenizeString(text))))
    vocab_subset = dict(zip(tokens, range(len(tokens))))
    end_time = labdate.MillisecondsTimestamp()
    # Return a new atomizer using the subset vocabulary.
    return WordAtomizer(vocab_subset)

class MaskLMAtomizer(AtomizerBase):
  """MaskLM corpus atomizer, as implemented in BERT model."""

  def __init__(self, 
               vocab: typing.Dict[str, int],
               metaTokens: typing.Dict[str, str],
               wordpiece_tokenization: bool
               ):
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.__init__()")
    self.wordpiece_tokenization = wordpiece_tokenization
    self.metaTokens = metaTokens
    super(MaskLMAtomizer, self).__init__(vocab)

  @classmethod
  def FromText(cls, 
               text: str,
               wordpiece_tokenization: bool
               ) -> "MaskLMAtomizer":
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus.

    Returns:
      An atomizer instance.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.FromText()")
    
    metaTokens = {
        '[START]'   : '[START]',
        '[END]'     : '[END]',
        '[PAD]'     : '[PAD]',
        '[MASK]'    : '[MASK]',
        '[HOLE]'    : '[HOLE]',
        '[ENDHOLE]' : '[ENDHOLE]',
    }

    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    atoms, _ = zip(*count_pairs)
    atoms = tuple(metaTokens.keys()) + atoms
    vocab = dict(zip(atoms, range(len(atoms))))

    return MaskLMAtomizer(vocab, metaTokens, wordpiece_tokenization)

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.AtomizeString()")

    encoded = []
    skipNext = 0
    for idx, char in enumerate(text):
      if skipNext > 0:
        skipNext -= 1
        continue
      try:
        if char == '[':
          for meta in self.metaTokens.values():
            if text[idx: idx + len(meta)] == meta:
              encoded.append(self.vocab[meta])
              skipNext = len(meta) - 1
              break
        if skipNext == 0:
          encoded.append(self.vocab[char])
      except KeyError:
        raise ValueError("Out of vocabulary word!")
    return np.array(encoded, dtype=np.int32)
    
  def __repr__(self) -> str:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.__repr__()")
    return f"MaskLMAtomizer[{self.vocab_size} chars]"
  
  @property
  def startToken(self):
    return self.vocab[self.metaTokens['[START]']]

  @property
  def endToken(self):
    return self.vocab[self.metaTokens['[END]']]

  @property
  def maskToken(self):
    return self.vocab[self.metaTokens['[MASK]']]

  @property
  def holeToken(self):
    return self.vocab[self.metaTokens['[HOLE]']]

  @property
  def endholeToken(self):
    return self.vocab[self.metaTokens['[ENDHOLE]']]

  @property
  def padToken(self):
    return self.vocab[self.metaTokens['[PAD]']]

  @property
  def startLabel(self):
    return self.metaTokens['[START]']

  @property
  def endLabel(self):
    return self.metaTokens['[END]']

  @property
  def maskLabel(self):
    return self.metaTokens['[MASK]']

  @property
  def holeLabel(self):
    return self.metaTokens['[HOLE]']

  @property
  def endholeLabel(self):
    return self.metaTokens['[ENDHOLE]']

  @property
  def padLabel(self):
    return self.metaTokens['[PAD]']
