# -*- coding: utf-8 -*-
"""
    poc 3
    PyInstallerで生成される実行ファイルのサイズ確認用
    ◇opencvを使用
"""


from pathlib import Path
# library
import numpy as np
import cv2

def read_file(file_name: str):
    p = Path(file_name)
    with p.open('rb') as file:
        return file.read()


class ImageData(object):
    """
        cv2.adaptiveThresholdの元データに使用するグレースケール画像を管理する。
    """
    def __init__(self, file_name: str):
        """
        :param file_name: 画像ファイル名
        """
        self.__file_name = file_name
        self.__gray_scale = ImageData.imread(file_name, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def imread(file_name: str, flags: int = cv2.IMREAD_COLOR):
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

            buffer = np.asarray(bytearray(read_file(file_name)), dtype=np.uint8)
            image = cv2.imdecode(buffer, flags)
        except FileNotFoundError as ex:
            # cv2.imread compatible
            pass
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
            pass
        return False

    @staticmethod
    def encode2PNM(image, flags: int):
        """
        読み込んだ画像データをPGMまたはPPM形式にエンコードする。
        :param image:
        :param flags:
        :return:
        """
        ext = {cv2.IMREAD_GRAYSCALE: '.PGM', cv2.IMREAD_COLOR: '.PPM'}.get(flags)
        assert ext
        ret_val, enc_img = cv2.imencode(ext, image, None)
        assert ret_val, 'encode2PNM failure'
        return enc_img.tobytes()

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
        return ImageData.imread(self.file_name)

    @property
    def gray_scale(self):
        """
            グレースケール画像(np.array)
        """
        assert self.__gray_scale is not None, 'cannot open file as image.'
        return self.__gray_scale


def main():
    image_file = r'../images/kodim07.png'
    data = ImageData(image_file)
    print(type(data.color))
    cv2.imshow('image', data.color)
    cv2.waitKey()


if __name__ == "__main__":
    main()
