# -*- coding: utf-8 -*-
"""
    Reporter
    ログハンドラーが重複登録されるのを防ぐために1箇所で生成してログハンドラーを返す。
"""
from logging import Logger, getLogger, DEBUG, StreamHandler

_reporters = []


def get_current_reporter() -> Logger:
    return _reporters[-1]


def __make_reporter(name: str='AdaptiveThreshold'):
    """
    @see https://docs.python.jp/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    :param name:
    :return:
    """
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    _reporters.append(logger)


__make_reporter()


def main():
    """
        Entry Point
    """
    logger = get_current_reporter()
    logger.debug("main")


if __name__ == "__main__":
    main()
