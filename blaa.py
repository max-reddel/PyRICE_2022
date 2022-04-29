import logging
import time
from tqdm import tqdm
import io


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, level=None):

        super(TqdmToLogger, self).__init__()

        logging.basicConfig(format='[blaa] %(message)s')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


if __name__ == "__main__":

    tqdm_out = TqdmToLogger()

    for x in tqdm(range(100), file=tqdm_out, mininterval=1, ):
        time.sleep(.1)
