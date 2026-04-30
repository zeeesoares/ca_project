import logging
import json
import time


def setup_log_metrics(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create file handler
    handler = logging.FileHandler(log_file)

    # Set JSON format
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name,
            }
            return json.dumps(log_record)
    
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger