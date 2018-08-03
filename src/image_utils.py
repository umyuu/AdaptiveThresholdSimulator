# -*- coding: utf-8 -*-
"""
    image_utils
    画像ファイルをasyncioを使って読み取ったりするモジュール
"""
import asyncio
#from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from functools import partial
from pathlib import Path
from typing import overload
# library
import numpy as np
import cv2
#
from reporter import get_current_reporter
from stopwatch import stop_watch

LOGGER = get_current_reporter()
ct = stop_watch()


def read_file(file_name: str):
    p = Path(file_name)
    with p.open('rb') as file:
        return file.read()


@overload
def memmap(file_name: Path, flags: int):
    pass


@overload
def memmap(file_name: str, flags: int):
    pass


def memmap(file_name, flags: int):
    """
    画像ファイルをメモリマップドファイルで読み込む。
    :Todo
    :param file_name:読み込みファイル名
    :param flags:cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE
    :return:
    """
    LOGGER.info("file_name:%s", file_name)
    mm = np.memmap(file_name, dtype=np.uint8, mode='r')
    image = cv2.imdecode(mm, flags)
    del mm  # メモリマップドファイルの割当を解除。
    return image


class ImageData(object):
    """
        cv2.adaptiveThresholdの元データに使用するグレースケール画像を管理する。
    """
    @overload
    def __init__(self, file_name: str):
        """
        オーバーロード定義、実体は@overloadアノテーションなしの関数
        変数に値を代入しているのは、IDEの警告抑止のため
        :param file_name:
        """
        self.__file_name = file_name
        self.task = None

    @overload
    def __init__(self, file_name: Path):
        self.__file_name = file_name
        self.task = None

    def __init__(self, file_name):
        """
        :param file_name: 画像ファイル名
        """
        self.__file_name = file_name
        self.__gray_scale = None
        LOOP = asyncio.get_event_loop()
        # タスクを登録するだけ、実際に動くのはrun_until_completeの呼び出し時。
        self.task = LOOP.create_task(ImageData.imread(file_name, cv2.IMREAD_GRAYSCALE))

    @staticmethod
    async def imread(file_name, flags: int = cv2.IMREAD_COLOR):
        """
        Unicode Path/Filename for imread Not supported.
        @see https://github.com/opencv/opencv/issues/4292
        cv2.imread alternative = np.asarray & cv2.imdecode
        Unicode Path/Filename image file read.

        :param file_name:画像ファイル名
        :param flags: cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE
        :return: {Mat}image BGR
            FileNotFoundError image is None
        """
        LOGGER.info("file_name:%s", ','.join([file_name, str(flags)]))
        loop = asyncio.get_event_loop()
        image = None
        try:
            ct()
            # IOブロッキングなので、run_in_executorにプールを指定して、処理を行う。
            # @see
            #   https://docs.python.jp/3/library/asyncio-dev.html#handle-blocking-functions-correctly
            #   http://iuk.hateblo.jp/entry/2017/01/27/173449
            #   https://djangostars.com/blog/asynchronous-programming-in-python-asyncio/
            image = await loop.run_in_executor(None, partial(memmap, file_name, flags))
            ct()
        except FileNotFoundError as ex:
            # cv2.imread compatible
            LOGGER.exception(ex)
        return image

    @staticmethod
    def imwrite(file_name, image, params=None) -> bool:
        """
        Unicode Path/Filename for imwrite Not supported.
        cv2.imwrite alternative = cv2.imencode & numpy.ndarray.tofile

        :param file_name:出力ファイル名 拡張子がエンコード形式
        :param image image 画像データ
        :param params encode params
        """
        try:
            if isinstance(file_name, Path):
                path = file_name
            else:
                path = Path(file_name)
            ret_val, buf = cv2.imencode(path.suffix, image, params)
            assert ret_val, 'imencode failure'
            with path.open('wb') as out:
                buf.tofile(out)
            return True
        except IOError as ex:
            LOGGER.exception(ex)
        return False

    @staticmethod
    def encode2PNM(image, ext: str):
        """
        読み込んだ画像データをPGMまたはPPM形式にエンコードする。
        :param image:
        :param flags:
        :return:
        """
        assert ext
        ret_val, enc_img = cv2.imencode(ext, image, None)
        assert ret_val, 'encode2PNM failure'
        b = enc_img.tobytes()
        print(len(b))
        return b

    @property
    def file_name(self) -> str:
        """
            画像ファイル名
        """
        if isinstance(self.__file_name, Path):
            return str(self.__file_name)
        else:
            return self.__file_name


    @property
    def color(self):
        """
            カラー画像(np.array)
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(loop.create_task(ImageData.imread(self.file_name)))

    @property
    def gray_scale(self):
        """
            グレースケール画像(np.array)
        """
        loop = asyncio.get_event_loop()
        self.__gray_scale = loop.run_until_complete(self.task)
        assert self.__gray_scale is not None, 'cannot open file as image.'
        return self.__gray_scale


def main():
    """
        Entry Point
    """
    with PoolExecutor(4) as io_pool:
        loop = asyncio.get_event_loop()
        loop.set_default_executor(io_pool)
        for file_name in [r'../images/kodim07.png', r'../images/kodim22.png']:
            image = ImageData(file_name)
            LOGGER.info(type(image.color))
            LOGGER.info(type(image.gray_scale))
        # shutdown_asyncgens Python 3.6
        if hasattr(loop, 'shutdown_asyncgens'):
            loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
