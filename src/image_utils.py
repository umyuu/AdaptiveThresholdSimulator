# -*- coding: utf-8 -*-
"""
    image_utils
    画像ファイルをasyncioを使って読み取ったりするモジュール
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
#from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from functools import partial
from reporter import get_current_reporter
from pathlib import Path

# library
import numpy as np
import cv2

LOGGER = get_current_reporter()

# get_event_loopではなくnew_event_loop
# Todo:contextlib.closingを使う。
LOOP = asyncio.get_event_loop()
#LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)
# Todo:初回起動の処理時間のボトルネックは↓のプロセスプール作成処理
LOOP.set_default_executor(PoolExecutor(4))


def stop_watch():
    from time import perf_counter
    from traceback import extract_stack
    from itertools import count
    start_time = perf_counter()
    endt_time = start_time
    c = count()

    def step():
        nonlocal endt_time
        func_name = extract_stack(None, 2)[0][2]
        n = next(c)
        elapsed = endt_time - start_time
        end_time = perf_counter()
        endt_time = end_time
        MSG = [func_name, n, end_time, elapsed]
        LOGGER.debug(MSG)
        return MSG, end_time - start_time

    return step


ct = stop_watch()


def read_file(file_name: str):
    p = Path(file_name)
    with p.open('rb') as file:
        return file.read()


def memmap(file_name: str, flags: int):
    """
    画像ファイルを読み込み、
    :param file_name:
    :param flags:
    :return:
    """
    mm = np.memmap(file_name, dtype=np.uint8, mode='r')
    image = cv2.imdecode(mm, flags)
    del mm  # メモリマップドファイルの割当を解除。
    return image


class ImageData(object):
    """
        cv2.adaptiveThresholdの元データに使用するグレースケール画像を管理する。
    """
    def __init__(self, file_name: str):
        """
        :param file_name: 画像ファイル名
        """
        self.__file_name = file_name
        #file_name ='a'
        self.__gray_scale = None
        ct()
        # スケジューリング
        self.task = LOOP.create_task(ImageData.imread(file_name, cv2.IMREAD_GRAYSCALE))
        ct()
        import time
        time.sleep(0)

    @staticmethod
    async def imread(file_name: str, flags: int = cv2.IMREAD_COLOR):
        """
        Unicode Path/Filename for imread Not supported.
        @see https://github.com/opencv/opencv/issues/4292
        cv2.imread alternative = np.asarray & cv2.imdecode
        Unicode Path/Filename image file read.
        :param file_name:
        :param flags: cv2.IMREAD_COLOR
        :return: {Mat}image BGR
            FileNotFoundError image is None
        """
        image = None
        try:
            ct()
            image = await LOOP.run_in_executor(None, partial(memmap, file_name, flags))
            ct()
        except FileNotFoundError as ex:
            # cv2.imread compatible
            LOGGER.exception(ex)
        return image

    @staticmethod
    def imwrite(file_name: str, image, params=None) -> bool:
        """
        Unicode Path/Filename for imwrite Not supported.
        cv2.imwrite alternative = cv2.imencode & numpy.ndarray.tofile
        :param file_name
        :param image imagedata
        :param params encode params
        """
        try:
            p = Path(file_name)
            ret_val, buf = cv2.imencode(p.suffix, image, params)
            assert ret_val, 'imencode failure'
            with p.open('wb') as out:
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
        return self.__file_name

    @property
    def color(self):
        """
            カラー画像(np.array)
        """
        return LOOP.run_until_complete(LOOP.create_task(ImageData.imread(self.file_name)))

    @property
    def gray_scale(self):
        """
            グレースケール画像(np.array)
        """
        self.__gray_scale = LOOP.run_until_complete(self.task)
        assert self.__gray_scale is not None, 'cannot open file as image.'
        return self.__gray_scale


def main():
    """
        Entry Point
    """
    pass


if __name__ == "__main__":
    main()
