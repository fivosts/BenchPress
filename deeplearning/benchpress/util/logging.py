# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tailor made logging module."""
import logging
import typing

from eupy.hermes import client

_logger = None

NOTSET   = logging.NOTSET
DEBUG    = logging.DEBUG
INFO     = logging.INFO
WARNING  = logging.WARNING
ERROR    = logging.ERROR
CRITICAL = logging.CRITICAL

PURPLE    = "\033[95m"
CYAN      = "\033[96m"
DARKCYAN  = "\033[36m"
BLUE      = "\033[94m"
GREEN     = "\033[92m"
YELLOW    = "\033[93m"
RED       = "\033[91m"
BOLD      = "\033[1m"
UNDERLINE = "\033[4m"
END       = "\033[0m"

def purple(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(PURPLE, string, END)

def cyan(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(CYAN, string, END)

def darkcyan(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(DARKCYAN, string, END)

def blue(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(BLUE, string, END)

def green(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(GREEN, string, END)

def yellow(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(YELLOW, string, END)

def red(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(RED, string, END)

def bold(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(BOLD, string, END)

def underline(string, callback = None):
  if callback:
    string = callback[0](string, callback = [x for x in callback if x != callback[0]])
  return "{}{}{}".format(UNDERLINE, string, END)

def output(string, *args):
  if args:
    string = args[0](string, callback = [x for x in args if x != args[0]])
  return string

class Logger:
  """
  Logger class API.
  """
  def __init__(self,
               name        : str,
               level       : int,
               mail_client : client.gmail,
               rank        : int,
               ):
    self.mail_client = mail_client
    self.rank = rank
    self.configLogger(name, level)
    return

  def configLogger(self, name, level):
    # create logger
    logging.root.handlers = []
    self.log = logging.getLogger(name)
    self.log.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    if not self.log.handlers:
      self.log.addHandler(ch)
      self.log.propagate = False
    self.info("Logger has been initialized")
    return

  @property
  def handlers(self):
    return self.log.handlers

  @property
  def logger(self):
    return self.log
  
  @property
  def level(self):
    return logging.getLevelName(self.log.level)

  @level.setter
  def level(self, lvl):
    self.log.setLevel(lvl)
    self.handlers[0].setLevel(lvl)
    return

  """
  Main logging functions
  """
  def debug(self,
            message   : str,
            color     : bool = True,
            ddp_nodes : bool = False
            ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, green)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.debug(message)
    return

  def info(self,
           message   : str,
           color     : bool = True,
           ddp_nodes : bool = False
           ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, cyan)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.info(message)
    return

  def warning(self,
              message   : str,
              color     : bool = True,
              ddp_nodes : bool = False
              ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, yellow)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.warning(message)
    return

  def warn(self,
           message   : str,
           color     : bool = True,
           ddp_nodes : bool = False
           ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, yellow)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.warn(message)
    return

  def error(self,
            message   : str,
            color     : bool = True,
            ddp_nodes : bool = False
            ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, red)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.error(message)
    return

  def critical(self,
               message   : str,
               color     : bool = True,
               ddp_nodes : bool = False
               ) -> None:
    if self.rank == 0 or ddp_nodes:
      if ddp_nodes:
        message = "N{}: {}".format(self.rank, message)
      if color:
        message = output(message, bold, underline, red)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.log.critical(message)
    return

  def shutdown(self):
    logging.shutdown()
    return

def logger() -> Logger:
  global _logger
  if not _logger:
    initLogger()
  return _logger

def initLogger(name = "", lvl = logging.INFO, mail = None, rank = 0):
  global _logger
  _logger = Logger(name, lvl, mail, rank)
  return _logger
