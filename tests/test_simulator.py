# -*- coding: utf-8 -*-
"""
    unittest
"""
from src.simulator import main


def test_pref_startup():
    """
    アプリ起動時のウィンドウ表示までの時間
    :return:
    """
    _, finish_time = main(False)
    assert finish_time < 0.3

def test_pref_startup():
    """
    アプリ起動時のウィンドウ表示までの時間
    :return:
    """
    _, finish_time = main(False)
    assert finish_time < 0.3
