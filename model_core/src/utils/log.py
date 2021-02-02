import logging
import threading
import inspect
from logging.handlers import TimedRotatingFileHandler


# 获得线程锁
lock = threading.Lock()
# 初始化日志对象为None
logger = None
# 初始化是否输出日志头为是
need_log_head = True


def logger_config(log_file_name='runtime.log', log_level='INFO', need_loghead=True, timed_rotating=False):
    global lock
    global logger
    global need_log_head

    if not logger:
        with lock:
            if not logger:
                logger = logging.getLogger()

                # 设置日志等级
                level = logging.INFO
                if log_level.lower() == 'info':
                    logger.setLevel(level=logging.INFO)
                if log_level.lower() == 'debug':
                    logger.setLevel(level=logging.DEBUG)

                # 设置日志文件存储方式
                if timed_rotating:
                    handler = TimedRotatingFileHandler(filename=log_file_name, when='D', interval=1, backupCount=3)
                else:
                    handler = logging.FileHandler(log_file_name)

                # 设置日志文件格式
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                need_log_head = need_loghead

    return logger


class LoggerHelper:
    global logger

    @staticmethod
    def info(log_text):
        """
        输出信息日志
        :param log_text:
        :return:
        """
        logging.info(LoggerHelper.__get_log_head() + str(log_text))

    @staticmethod
    def warn(log_text):
        """
        输出警告日志
        :param log_text:
        :return:
        """
        logging.warn(LoggerHelper.__get_log_head() + str(log_text))

    @staticmethod
    def debug(log_text):
        """
        输出调试日志
        :param log_text:
        :return:
        """
        logging.debug(LoggerHelper.__get_log_head() + str(log_text))

    @staticmethod
    def error(log_text):
        """
        输出错误日志
        :param log_text:
        :return:
        """
        logging.error(LoggerHelper.__get_log_head() + str(log_text))

    @staticmethod
    def __get_log_head():
        """
        获取日志头
        :return:
        """
        invoke_file = LoggerHelper.__get_file_name(inspect.stack()[1][1])
        invoke_line = inspect.stack()[1][2]
        log_head = "%s:%d " % (invoke_file, invoke_line) if need_log_head else ""
        return log_head

    @staticmethod
    def __get_file_name(file_path):
        """
        获取输出日志的文件名
        :param file_path:
        :return:
        """
        data = file_path.strip().strip("/").split("/")
        return data[-1]


# 仅供单元测试
if __name__ == '__main__':
    print(logger_config())
