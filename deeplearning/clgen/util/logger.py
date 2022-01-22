import logging
import typing

from deeplearning.clgen.util import environment

from eupy.hermes import client

logger = None

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
               colorize    : bool,
               step        : bool,
               ):
    self.mail_client = mail_client
    self.colorize    = colorize
    self.step        = step
    self.configLogger(name, level)
    return

  def _configLogger(self, name, level):
    # create logger
    logging.root.handlers = []
    self.logger = logging.getLogger(name)
    self.logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    if not self.logger.handlers:
      self.logger.addHandler(ch)
      self.logger.propagate = False
    self.info("Logger has been initialized")
    return

  @property
  def handlers(self):
    return self.logger.handlers

  @property
  def logger(self):
    return self.logger
  
  @property
  def level(self):
    return logging.getLevelName(self.logger.level)

  @level.setter
  def level(self, lvl):
    self.logger.setLevel(lvl)
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
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, green)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.debug(message)
    return

  def info(self,
          message   : str,
          color     : bool = True,
          ddp_nodes : bool = False
          ) -> None:
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, cyan)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.info(message)
    return

  def warning(self,
              message   : str,
              color     : bool = True,
              ddp_nodes : bool = False
              ) -> None:
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, yellow)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.warning(message)
    return

  def warn(self,
           message   : str,
           color     : bool = True,
           ddp_nodes : bool = False
           ) -> None:
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, yellow)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.warn(message)
    return

  def error(self,
            message   : str,
            color     : bool = True,
            ddp_nodes : bool = False
            ) -> None:
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, red)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.error(message)
    return
  
  def critical(self,
               message   : str,
               color     : bool = True,
               ddp_nodes : bool = False
               ) -> None:
    if environment.WORLD_RANK == 0 or ddp_nodes:
      if color:
        message = output(message, bold, underline, red)
      if self.mail_client:
        self.mail_client.send_message("Logger", message)
      self.logger.critical(message)
    return

  def shutdown(self):
    logging.shutdown()
    return

def initLogger(name, lvl = logging.INFO, mail = None):
  global logger
  logger = Logger(name, lvl, mail)
  return logger
