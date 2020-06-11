import logging


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

        grey = "\x1b[38;21m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        light_blue = "\x1b[1;36m"

        self.__formats = {
            logging.DEBUG: light_blue + self._fmt + reset,
            logging.INFO: grey + self._fmt + reset,
            logging.WARNING: yellow + self._fmt + reset,
            logging.ERROR: red + self._fmt + reset,
            logging.CRITICAL: bold_red + self._fmt + reset
        }

    def format(self, record):
        log_fmt = self.__formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggingMixin:
    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        logger = logging.getLogger(name)
        if len(logger.handlers) == 0:
            handler = logging.StreamHandler()
            formatter = CustomFormatter('%(levelname)-5s | %(name)s | %(asctime)s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger
