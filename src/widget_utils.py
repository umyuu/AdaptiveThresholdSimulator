# -*- coding: utf-8 -*-
"""
    Widget Utils

"""
# GUI
import tkinter as tk
import tkinter.ttk as ttk
# Library
import cv2
#
from image_utils import ImageData
from reporter import get_current_reporter
from stopwatch import stop_watch

LOGGER = get_current_reporter()
ct = stop_watch()


class WidgetUtils(object):
    """
        ウィジット用のユーティリティ
    """
    @staticmethod
    def bind_all(widget: tk.Widget, modifier: str = "", letter: str = "", callback=None) -> None:
        """
        Keyboard Shortcut Assign.
        :param widget:割り当てるウィジット
        :param modifier:キーコード修飾子 Ctrl, Shift, Altなど
        :param letter:割当文字
        :param callback:イベント発生時に呼び出されるコールバック
        :return:
        """
        widget.bind_all('<{0}-{1}>'.format(modifier, letter.upper()), callback)
        # numeric letter multi assign check.
        if not letter.isdecimal():
            widget.bind_all('<{0}-{1}>'.format(modifier, letter.lower()), callback)

    @staticmethod
    def set_visible(widget: tk.Widget, visible: bool = False) -> None:
        """
            ウィンドウの表示/非表示を行う。
        """
        if visible:
            widget.deiconify()
        else:
            widget.withdraw()

    @staticmethod
    def update_image(widget: tk.Widget, img, encode_mode: int=cv2.IMREAD_GRAYSCALE) -> None:
        """
            表示画像の更新を行う。
            :param widget 画像を表示対象
            :param img イメージデータ
            :param encode_mode 出力画像形式　カラー / グレースケール

            note:ImageData.encode2PNMを使っている理由はexeファイルのサイズ削減のため。
                Pillowに依存しないことで1MB、実行ファイルサイズが削減されます。
        """
        assert img is not None
        # 画像出力用
        widget.np = img
        ct()
        ext = {cv2.IMREAD_GRAYSCALE: '.PGM', cv2.IMREAD_COLOR: '.PPM'}.get(encode_mode)
        # PGMまたはPBM形式に変換する。
        # GC対象にならないように参照を保持
        #ext = ".png"
        widget.src = tk.PhotoImage(data=ImageData.encode2PNM(img, ext))
        ct()
        widget.configure(image=widget.src)
        ct()


class ImageWindow(tk.Toplevel):
    """
    カラー画像とグレースケール画像を表示するために、別ウィンドウとする。
    """
    def __init__(self, master=None, cnf: dict = None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)
        self.protocol('WM_DELETE_WINDOW', self.on_window_exit)
        # packでウィンドウが表示されるので、初期表示は非表示に。
        WidgetUtils.set_visible(self, False)
        self.__label_image = tk.Label(self)
        self.__label_image.pack()
        self.__var = None

    def on_window_exit(self):
        WidgetUtils.set_visible(self, False)
        self.var.set(False)

    def update_image(self, img, read_mode: int):
        WidgetUtils.update_image(self.__label_image, img, read_mode)

    @property
    def tag(self) -> int:
        """
            タグ(Getter)
        """
        return int(self.data_attributes['data-tag'])

    @property
    def var(self) -> tk.BooleanVar:
        """
            BooleanVar(Getter)
        """
        return self.__var

    @var.setter
    def var(self, value: tk.BooleanVar):
        """
            タグ(Getter)
        """
        self.__var = value


class ScrollListBox(tk.Listbox):
    """
    スクロールバー対応のListbox
    """
    def __init__(self, master=None, cnf: dict = None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)
        self.y_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.yview)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(yscrollcommand=self.y_scrollbar.set)
        self.x_scrollbar = tk.Scrollbar(master, orient=tk.HORIZONTAL, command=self.xview)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.configure(xscrollcommand=self.x_scrollbar.set)


class ScrollTreeview(ttk.Treeview):
    """
    スクロールバー対応のTreeview
    """
    def __init__(self, master=None, cnf: dict = None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)
        self.y_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.yview)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(yscrollcommand=self.y_scrollbar.set)
        self.pack(side=tk.LEFT, fill=tk.Y)
        self.x_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.xview)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.configure(yscrollcommand=self.x_scrollbar.set)


class SplashScreen(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)


def main():
    """
        Entry Point
    """
    pass


if __name__ == "__main__":
    main()
