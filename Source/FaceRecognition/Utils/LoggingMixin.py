import logging


class LoggingMixin:
    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        logger = logging.getLogger(name)
        if len(logger.handlers) == 0:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)-5s | %(name)s | %(asctime)s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger
