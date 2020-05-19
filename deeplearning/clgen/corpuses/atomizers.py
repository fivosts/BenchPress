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

from deeplearning.clgen import errors
from labm8.py import app
from labm8.py import labdate

from eupy.native import logger as l

FLAGS = app.FLAGS


class AtomizerBase(object):
  """The base class for implementing atomizers."""

  def __init__(self, vocab: typing.Dict[str, int]):
    """Instantiate an atomizer.

    Args:
      vocab: A dictionary of mappings from character sequences (atoms) into
        indices.

    Raises:
      TypeError: If vocab is not a dictionary.
      InvalidVocab: If the dictionary of mappings includes any duplicate values.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AtomizerBase.__init__()")
    self.vocab = vocab
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
      raise errors.InvalidVocab("all atoms must be unique")
    if len(set(self.vocab.values())) != len(self.vocab):
      raise errors.InvalidVocab("all indices must be unique")

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
        raise UserError("Wrong encoded array specified")
    except KeyError:
      raise ValueError

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

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.AtomizeString()")
    try:
      return np.array(list(map(lambda x: self.vocab[x], text)), dtype=np.int32)
    except KeyError:
      raise ValueError

  def __repr__(self) -> str:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.__repr__()")
    return f"AsciiCharacterAtomizer[{self.vocab_size} chars]"

  @classmethod
  def FromText(cls, text: str) -> "AsciiCharacterAtomizer":
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus.

    Returns:
      An atomizer instance.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.AsciiCharacterAtomizer.FromText()")
    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    atoms, _ = zip(*count_pairs)
    vocab = dict(zip(atoms, range(len(atoms))))
    return AsciiCharacterAtomizer(vocab)


class GreedyAtomizer(AtomizerBase):
  """A greedy atomizer supports multi-character tokens."""

  def __init__(self, vocab: typing.Dict[str, int], determine_chars=False):
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.GreedyAtomizer.__init__()")
    self.determine_chars = determine_chars
    super(GreedyAtomizer, self).__init__(vocab)

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
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.GreedyAtomizer.AtomizeString()")

    def _AddToVocab(token: str) -> int:
      """Add a token to the vocabulary and return its index."""
      l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.GreedyAtomizer._AddToVocab()")
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
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.GreedyAtomizer.__repr__()")
    return f"GreedyAtomizer[{self.vocab_size} tokens]"

  @classmethod
  def FromText(cls, text: str, atoms: typing.Set[str]) -> "GreedyAtomizer":
    """Instantiate and an atomizer from a corpus text.

    Args:
      text: Text corpus
      atoms: A list of multi-character tokens.

    Returns:
      An atomizer instance.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.GreedyAtomizer.FromText()")
    if not atoms:
      raise ValueError("No atoms specified")

    # Instantiate a greedy atomizer using the full vocabulary.
    full_vocab = dict(zip(atoms, range(len(atoms))))
    c = GreedyAtomizer(full_vocab, determine_chars=True)
    # Derive the subset of the vocabulary required to encode the given text.
    tokens = sorted(list(set(c.TokenizeString(text))))
    vocab_subset = dict(zip(tokens, range(len(tokens))))
    end_time = labdate.MillisecondsTimestamp()
    # Return a new atomizer using the subset vocabulary.
    return GreedyAtomizer(vocab_subset)

class MaskLMAtomizer(AtomizerBase):
  """MaskLM corpus atomizer, as implemented in BERT model."""

  def __init__(self, 
               vocab: typing.Dict[str, int],
               wordpiece_tokenization: bool
               ):
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.__init__()")
    self.wordpiece_tokenization = wordpiece_tokenization
    self._maskLabel = "[MASK]"
    super(MaskLMAtomizer, self).__init__(vocab)

  def AtomizeString(self, text: str) -> np.array:
    """Atomize a text into an array of vocabulary indices.

    Args:
      text: Input text.

    Returns:
      An array of indices into vocabulary for all atoms in text.
    """
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.AtomizeString()")
    try:
      return np.array(list(map(lambda x: self.vocab[x], text)), dtype=np.int32)
    except KeyError:
      raise ValueError

  def __repr__(self) -> str:
    l.getLogger().debug("deeplearning.clgen.corpuses.atomizers.MaskLMAtomizer.__repr__()")
    return f"MaskLMAtomizer[{self.vocab_size} chars]"

  @property
  def maskToken(self):
    return self.vocab[self._maskLabel]

  @property
  def maskLabel(self):
    return self._maskLabel
  
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
    
    ## Account for MetaTokens. CLS and SEP or START might need to be added.
    metaTokens = ('[MASK]',)

    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    atoms, _ = zip(*count_pairs)
    atoms = metaTokens + atoms
    vocab = dict(zip(atoms, range(len(atoms))))

    return MaskLMAtomizer(vocab, wordpiece_tokenization)
