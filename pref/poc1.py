# -*- coding: utf-8 -*-
"""
    poc 1
    非同期画像読み込み
"""
import asyncio
from contextlib import closing
from functools import partial
from threading import get_ident
import time
from os import getpid
from pathlib import Path
from random import choice

from concurrent.futures import ThreadPoolExecutor as PoolExecutor
#from concurrent.futures import ProcessPoolExecutor as PoolExecutor


def log(kwargs):
    msg = "Process ID=%d, Thread ID=%d,%s" % (getpid(), get_ident(), ', '.join(map(str, kwargs)))
    print(msg)


def read_file(filename: str):
    p = Path(filename)
    with p.open('rb') as f:
        time.sleep(choice([3, 5]))
        log(['end read_file', filename])
        return f.read()


async def imread(filename : str):
    loop = asyncio.get_event_loop()
    log(['begin imread', loop.time(), filename])
    return await loop.run_in_executor(None, partial(read_file, filename))


async def imread2(filename : str):
    """
    note:thread blocking.
    :param filename:
    :return:
    """
    loop = asyncio.get_event_loop()
    log(['begin imread2', loop.time(), filename])
    p = Path(filename)
    with p.open('rb') as f:
        time.sleep(choice([3, 5]))
        log(['end read_file', filename])
        return f.read()


def main():
    """
        Entry Point
        画像イメージを非同期で読み込む
    """
    with closing(asyncio.new_event_loop()) as LOOP, PoolExecutor(2) as FileIOPool:
        LOOP.set_debug(True)
        LOOP.set_default_executor(FileIOPool)
        log(['begin create_task', LOOP.time()])
        task1 = LOOP.create_task(imread(r'../images/kodim07.png'))
        task2 = LOOP.create_task(imread(r'../images/sakura.jpg'))
        log(['end create_task', LOOP.time()])

        time.sleep(3)
        log(['sleep', LOOP.time()])

        val1 = LOOP.run_until_complete(task1)
        val2 = LOOP.run_until_complete(task2)

        log(['Finished', LOOP.time()])


if __name__ == "__main__":
    main()
