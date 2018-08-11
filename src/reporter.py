# -*- coding: utf-8 -*-
"""
    Reporter
    ログハンドラーが重複登録されるのを防ぐために1箇所で生成してログハンドラーを返す。
"""
from logging import Logger, getLogger, Formatter, StreamHandler
from logging import DEBUG

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
    formatter = Formatter('%(asctime)s pid:%(process)05d, tid:%(thread)05d - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(DEBUG)
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    # asyncioのログレベルを変更
    # @see https://docs.python.org/3/library/asyncio-dev.html#logging
    getLogger('asyncio').setLevel(DEBUG)
    _reporters.append(logger)


__make_reporter()


def main():
    """
        Entry Point
    """
    assert len(_reporters) == 1
    logger = get_current_reporter()
    logger.debug("main")


if __name__ == "__main__":
    main()
