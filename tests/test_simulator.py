# -*- coding: utf-8 -*-
"""
    unittest
"""
import cv2
from hashlib import sha384
import pytest

# test target
from src.image_utils import ImageData, read_file
from src.simulator import main


def test_pref_startup(request):
    print(request)
    """
    アプリ起動時のウィンドウ表示までの時間
    :return:
    """
    _, finish_time = main(False)
    #assert finish_time < 0.3
    assert finish_time < 2


@pytest.fixture
def file_hash_result() -> dict:
    return {
        "color.png":
        "e9a8e38e1a16b0999b03b0d6fe01e985d04d7280c4a3052c2575694eb396103916947ab755b3a1df451295ef6e5dfe65",
        "gray_scale.png":
        "7772f2a319c3f4a9d5ed9846126616056e41b29d4dd1287e51c3e1cfebc842ade3a7d945e2544f5a4e6105d59568e853",
        "adaptiveThreshold.png":
        'b3b68097b9f589a74eb92433cd896b5dbba5676482034720f3b4e0b460749676b6c7aff2fca6b69de0665f1bd955f346'
    }


@pytest.fixture
def image_data() -> ImageData:
    return ImageData(r'../images/kodim07.png')


def test_hash_image_color(tmpdir, image_data: ImageData, file_hash_result: dict):
    """
    出力画像が変わらないことを保証するため。
    以下の3画像のファイルハッシュ値を算出する。
        カラー画像
        グレースケール画像
        adaptiveThreshold後の画像
    :param tmpdir:一時ファイル作成用
    :param image_data:
    :param file_hash:各画像ファイルのファイルハッシュ値
    :return:
    """
    name = "color.png"
    hash_result = file_hash_result[name]
    file = tmpdir.mkdir("images").join(name)
    assert ImageData.imwrite(file, image_data.color)
    assert sha384(read_file(file)).hexdigest() == hash_result


def test_hash_image_gray_scale(tmpdir, image_data: ImageData, file_hash_result: dict):
    """
    グレースケール画像出力し、ファイルハッシュ値を算出する。
    :param tmpdir:一時ファイル作成用
    :return:
    """
    name = "gray_scale.png"
    hash_result = file_hash_result[name]
    file = tmpdir.mkdir("images").join(name)
    print(file)
    assert ImageData.imwrite(file, image_data.gray_scale)
    assert sha384(read_file(file)).hexdigest() == hash_result


def test_hash_image_adaptiveThreshold(tmpdir, image_data: ImageData, file_hash_result: dict):
    """
    cv2.adaptiveThresholdの画像を出力し、ファイルハッシュ値を算出する。
    :param tmpdir:一時ファイル作成用
    :param
    :return:
    """
    name = "adaptiveThreshold.png"
    hash_result = file_hash_result[name]
    file = tmpdir.mkdir("images").join(name)
    params = 255, 1, 0, 11, 2
    result = cv2.adaptiveThreshold(image_data.gray_scale, *params)
    print(file)
    assert ImageData.imwrite(file, result)
    assert sha384(read_file(file)).hexdigest() == hash_result

