# -*- coding: utf-8 -*-
"""
    Resources
    リソースをロードするクラス。作りかけ！！
"""
from src.reporter import get_current_reporter
LOGGER = get_current_reporter()

_loaders = []


class Resources(object):
    def __init__(self):
        pass

    def load(self):
        pass

    def prefetch(self, file_name: str):
        print(file_name)
        pass


def get_current_loader() -> Resources:
    return _loaders[-1]


def __make_loader():
    res = Resources()
    _loaders.append(res)


__make_loader()


def main():
    """
        Entry Point
    """
    resource = get_current_loader()
    resource.prefetch("aaa")


if __name__ == "__main__":
    main()
