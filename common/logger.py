#  MIT License
#
#  Copyright (c) 2019 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""
Logging helper package.

Example:

    >>> mylogger = DefaultLogger('test_logger')
    >>> mylogger.logger.name
    'test_logger'

    >>> mylogger.logger
    <Logger test_logger (INFO)>

    >>> mylogger.info("test")
    asdf

"""

import logging
import sys

from typing import Iterable, Union
from neptune.experiments import Experiment

class DefaultLogger(object):
    """Default class for generating formatted log messages.

    Attributes:
        level (int): Logging level
    """

    def __init__(self, logger_name, level=logging.INFO):
        """Default logger initialization

        Args:
            logger_name (str): Name of the logger instance.
            level (int, optional): Logging level. Default `logging.INFO`
        """
        self._level: int = level
        self._logger: logging.Logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        self.__neptune_experiment = None

        handlers: Union[Iterable[logging.Handler], logging.Handler] = self._get_handlers()

        if isinstance(handlers, list):
            for handler in handlers:
                self._logger.addHandler(handler)
        else:
            self._logger.addHandler(handlers)

    def info(self, msg: str):
        """Stores an `INFO` level message.

        Args:
            msg (str): The message
        """
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def send_metric(self, channel_name, x, y=None, timestamp=None):

        if self.neptune_experiment is not None:
            self.neptune_experiment.send_metric(channel_name, x, y, timestamp)

    def debug(self, msg: str):
        """Stores a `DEBUG` level message.

        Args:
            msg (str): The message
        """
        self._logger.debug(msg)

    @property
    def neptune_experiment(self) -> Experiment:
        return self.__neptune_experiment

    @neptune_experiment.setter
    def neptune_experiment(self, experiment: Experiment):
        self.__neptune_experiment = experiment

    @property
    def logger(self) -> logging.Logger:
        """
        logging.Logger: Logger instance."""
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        self._logger = logger

    @property
    def level(self) -> int:
        """Returns the current logging level.

        >>> DefaultLogger('test_logger').level
        20

        >>> DefaultLogger('test_logger', logging.DEBUG).level
        10

        Returns:
            int: Logging level
        """
        return self._level

    @level.setter
    def level(self, level: int):
        if self._logger is not None:
            self._logger.setLevel(level)
        self._level = level

    def _get_handlers(self) -> Union[Iterable[logging.Handler], logging.Handler]:
        """Creates and returns the default `logging.StreamHandler` logging handler.

        Child objects can override this method and can creates its own handler.

        >>> DefaultLogger('test_logger')._get_handlers()
        <StreamHandler (INFO)>

        Returns:
            Logging handler or an iterable of handlers.
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.level)
        handler.setFormatter(self._get_logging_formatter())

        return handler

    @staticmethod
    def _get_logging_formatter():
        """Returns a `logging.Formatter` with specified message format.

        Returns:
            `logging.Formatter`
        """
        return logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                 datefmt='%Y-%m-%d %H-%M-%S')


class FileLogger(DefaultLogger):
    """
    A logger class which writes formatted logging messages to disk files.
    """

    def __init__(self, logger_name, output_dir, filename, extension='log',
                 level=logging.INFO, console=False):
        """

        Args:
            logger_name (str): Name of the logger instance.
            output_dir (str): Output directory for saving the log
            filename (str): Output filename
            extension (str, optional): Extension of the log file. Default: `log`
            level (int, optional): Logging level. Default: `logging.INFO`
            console (bool, optional): Print the messages to the console or not. Default: `False`
        """

        self.output_dir = output_dir
        self.filename = filename
        self.extension = extension
        self.console = console

        super().__init__(logger_name, level)

    def _get_handlers(self) -> Union[Iterable[logging.Handler], logging.Handler]:
        handlers = []

        if self.console:
            handlers.append(super()._get_handlers())

        file_handler = logging.FileHandler(self.output_dir + '/' + self.filename +
                                           '.' + self.extension)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._get_logging_formatter())

        handlers.append(file_handler)

        return handlers
