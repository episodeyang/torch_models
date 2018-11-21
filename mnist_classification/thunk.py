def thunk(fn, log_dir=None, log_prefix=None, ):
    """
    thunk for configuring the logger

    :param fn: function to be called
    :param log_dir: logging director for the logger
    :param log_prefix: logging prefix for the logger.
    :return: None
    """
    import traceback
    from ml_logger import logger
    logger.configure(log_directory=log_dir, prefix=log_prefix)
    logger.log_revision(True)
    logger.log_caller(fn)

    def _(*args, **kwargs):
        from ml_logger import logger
        logger.configure(log_directory=log_dir, prefix=log_prefix)
        host = dict(hostname=logger.hostname)
        logger.log_params(host=host)
        try:
            fn(*args, **kwargs)
            logger.log_line("=============== execution is complete ===============")
        except Exception as e:
            tb = traceback.format_exc()
            logger.log_line(tb)
            logger.flush()
            raise e

    return _
