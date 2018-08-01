# -*- coding: utf-8 -*-
"""
    AdaptiveThreshold Simulator
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
#from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from collections import deque
from datetime import datetime
from enum import Enum, auto
from functools import partial

from logging import getLogger, DEBUG, StreamHandler
from pathlib import Path
import sys
# gui
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
# library
import numpy as np
import cv2


PROGRAM_NAME = 'AdaptiveThreshold'
__version__ = '0.0.4'
# logging
HANDLER = StreamHandler()
HANDLER.setLevel(DEBUG)
LOGGER = getLogger(PROGRAM_NAME)
LOGGER.setLevel(DEBUG)
LOGGER.addHandler(HANDLER)

# get_event_loopではなくnew_event_loop
# contextlib.closingを使う。
LOOP = asyncio.get_event_loop()
#LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)
LOOP.set_default_executor(PoolExecutor(8))


def stop_watch():
    from time import perf_counter
    from traceback import extract_stack
    from itertools import count
    start_time = perf_counter()
    c = count()

    def step():
        func_name = extract_stack(None, 2)[0][2]
        n = next(c)
        end_time = perf_counter()
        MSG = [func_name, n, end_time]
        LOGGER.debug(MSG)
        return MSG, end_time - start_time

    return step


ct = stop_watch()


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
        #file_name ='a'
        self.__gray_scale = None
        ct()
        # スケジューリング
        #self.task = asyncio.ensure_future(ImageData.imread(file_name, cv2.IMREAD_GRAYSCALE),loop=LOOP)
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
            val = await LOOP.run_in_executor(None, partial(read_file, file_name))
            buffer = np.asarray(bytearray(val), dtype=np.uint8)
            image = cv2.imdecode(buffer, flags)
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
        return LOOP.run_until_complete(LOOP.create_task(ImageData.imread(self.file_name)))

    @property
    def gray_scale(self):
        """
            グレースケール画像(np.array)
        """
        self.__gray_scale = LOOP.run_until_complete(self.task)
        assert self.__gray_scale is not None, 'cannot open file as image.'
        return self.__gray_scale


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
        # GC対象にならないように参照を保持
        # PGMまたはPBM形式に変換する。
        widget.src = tk.PhotoImage(data=ImageData.encode2PNM(img, encode_mode))
        ct()
        widget.configure(image=widget.src)
        ct()


class ImageWindow(tk.Toplevel):
    """
    カラー画像とグレースケール画像を表示するために、別ウィンドウとする。
    """
    def __init__(self, master=None, tag: int=None, var: tk.BooleanVar=None):
        super().__init__(master)
        self.protocol('WM_DELETE_WINDOW', self.on_window_exit)
        # packでウィンドウが表示されるので、初期表示は非表示に。
        WidgetUtils.set_visible(self, False)
        self.__label_image = tk.Label(self)
        self.__label_image.pack()
        self.__tag = tag
        self.__var = var

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
        return self.__tag

    @property
    def var(self) -> tk.BooleanVar:
        """
            BooleanVar(Getter)
        """
        return self.__var

# 画像形式
ImageFormat = {
    'PNG': ('*.png', ),
    'JPEG': ('*.jpg', '*.jpeg', ),
    'WEBP': ('*.webp', ),
    'BMP': ('*.bmp', )
}

def askopenfilename(widget: tk.Widget) -> str:
    """
    ファイルを開くダイアログ
    :return:ファイルパス
    """
    IMAGE_FILE_TYPES = [('Image Files', ImageFormat['PNG'] + ImageFormat['JPEG'] + ImageFormat['WEBP'] + ImageFormat['BMP']),
                        ('png (*.png)', ImageFormat['PNG']),
                        ('jpg (*.jpg, *.jpeg)', ImageFormat['JPEG']),
                        ('webp (*.webp)', ImageFormat['WEBP']),
                        ('bmp (*.bmp)', ImageFormat['BMP']),
                        ('*', '*.*')]
    return filedialog.askopenfilename(parent=widget, filetypes=IMAGE_FILE_TYPES)


class SplashScreen(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)


class CONTROLS(Enum):
    ADAPTIVE = auto()
    THRESHOLD_TYPE = auto()
    BLOCK_SIZE = auto()
    C = auto()

class Panel(Enum):
    a_side = auto()
    main_side = auto()


class Application(tk.Frame):
    """
        Main Window
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.master.title('AdaptiveThreshold Simulator Ver:{0}'.format(__version__))
        self.master.update_idletasks()
        self.data = None  # type: ImageData
        self.component = {}
        self.a_side = tk.Frame(self)
        self.component[Panel.a_side] = self.a_side  # 左側のコンテンツ
        self.main_side = tk.Frame(self)
        self.component[Panel.main_side] = self.main_side  # 右側のコンテンツ
        self.top_frame = tk.LabelFrame(self.component[Panel.a_side], text='params')

        #self.a_side =
        #self.main_side = tk.Frame(self)  # 右側のコンテンツ
        # Data Bind Member
        self.var_file_name = tk.StringVar()
        self.var_creation_time = tk.StringVar()
        self.var_original = tk.BooleanVar(value=False)
        self.var_gray_scale = tk.BooleanVar(value=False)
        #
        self.color_image = ImageWindow(self, cv2.IMREAD_COLOR, self.var_original)
        self.gray_scale_image = ImageWindow(self, cv2.IMREAD_GRAYSCALE, self.var_gray_scale)
        self.history = deque(maxlen=12)
        self.menu_bar = self.create_menubar()
        self.master.configure(menu=self.menu_bar)
        self.create_widgets()
        self.component[Panel.a_side].pack(side=tk.LEFT, anchor=tk.NW)
        self.component[Panel.main_side].pack(side=tk.LEFT, expand=True, fill=tk.BOTH, anchor=tk.NW)

    def create_widgets(self):
        """
            メイン画面項目を生成/配置
            1,左側のコンテンツ  a_side
            1-1,入力パラメータ欄
            1-2,コマンド欄
            1-3,出力欄
            2,右側のコンテンツ  main_side
        """
        self.top_frame.pack(anchor=tk.NW)
        self.controls = dict()
        data = {CONTROLS.ADAPTIVE: (tk.Scale, self.top_frame, {
                    'label': '0:MEAN_C / 1:GAUSSIAN_C',
                    'from_': cv2.ADAPTIVE_THRESH_MEAN_C,
                    'to': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    'length': 300, 'orient': tk.HORIZONTAL}),
                CONTROLS.THRESHOLD_TYPE: (tk.Scale, self.top_frame, {
                    'label': '0:BINARY / 1:INV',
                    'from_': cv2.THRESH_BINARY, 'to': cv2.THRESH_BINARY_INV,
                    'length': 300, 'orient': tk.HORIZONTAL}),
                CONTROLS.BLOCK_SIZE: (tk.Scale, self.top_frame, {
                    'label': 'BLOCK_SIZE', 'from_': 3, 'to': 255,  # initial stepvalue 3.
                    'length': 300, 'orient': tk.HORIZONTAL}),
                CONTROLS.C: (tk.Scale, self.top_frame, {
                    'label': 'C', 'from_': 0, 'to': 255,
                    'length': 300, 'orient': tk.HORIZONTAL}),
                }

        for k, v in data.items():
            widget, parent, params = v
            self.controls[k] = widget(parent, params)

        self.scale_reset()
        # コマンドの登録処理
        # この位置で登録するのは self.draw イベントの発生を抑止するため。
        for child in self.top_frame.children.values():
            child.configure(command=self.draw)
            child.pack()

        self.command_frame = tk.Frame(self.a_side)
        self.button_reset = tk.Button(self.command_frame, text='RESET', command=self.scale_reset)
        self.button_reset.pack()
        self.command_frame.pack()

        self.create_output_frame()

        # create main side widget
        self.message_panel = tk.Label(self.main_side, text='Ctrl+S…Image Save Dialog')
        self.message_panel.pack(anchor=tk.NW)
        self.entry_filename = tk.Entry(self.main_side, textvariable=self.var_file_name)
        # fillで横にテキストボックスを伸ばす
        self.entry_filename.pack(anchor=tk.NW, fill=tk.X)
        self.entry_creation_time = tk.Entry(self.main_side, textvariable=self.var_creation_time)
        # fillで横にテキストボックスを伸ばす
        self.entry_creation_time.pack(anchor=tk.NW, fill=tk.X)

        #self.entry_filename.pack(anchor=tk.NW, expand=True, fill=tk.X)
        self.label_image = tk.Label(self.main_side)
        self.label_image.np = None
        self.label_image.pack(anchor=tk.NW, pady=10)
        #self.label_image.pack(anchor=tk.NW, fill=tk.BOTH)
        #self.label_image.pack(anchor=tk.NW, expand=True, fill=tk.BOTH)

    def create_output_frame(self):
        """
            パラメータ値の出力欄
        """
        self.output_frame = tk.LabelFrame(self.a_side, text='output')
        self.output_frame.pack(side=tk.TOP, fill=tk.Y)
        MSG = 'Select a row and Ctrl+C\nCopy it to the clipboard.'
        self.label_message = tk.Label(self.output_frame, text=MSG)
        self.label_message.pack(expand=True, side=tk.TOP, fill=tk.X)

        class ScrollListBox(tk.Listbox):
            """
            スクロールバー対応のリストボックス
            """
            def __init__(self, master=None, cnf: dict = None, **kw):
                if cnf is None:
                    cnf = {}
                super().__init__(master, cnf, **kw)
                self.y_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.yview)
                self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.configure(yscrollcommand=self.y_scrollbar.set)
                self.pack(side=tk.LEFT, fill=tk.Y)
        class ScrollTreeview(ttk.Treeview):
            """
            スクロールバー対応のリストボックス
            """
            def __init__(self, master=None, cnf: dict = None, **kw):
                if cnf is None:
                    cnf = {}
                super().__init__(master, cnf, **kw)
                self.y_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.yview)
                self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.configure(yscrollcommand=self.y_scrollbar.set)
                self.pack(side=tk.LEFT, fill=tk.Y)

        #self.listbox = ScrollListBox(self.output_frame, width=40, height=self.history.maxlen)
        self.listbox = ScrollListBox(self.output_frame, width=40, height=self.history.maxlen)

    def create_menubar(self) -> tk.Menu:
        """
        MenuBarの作成
        :return:
        """
        menu_bar = tk.Menu(self, tearoff=False)

        def crate_file_menu() -> tk.Menu:
            """
            ファイルメニュー
            """
            menu = tk.Menu(self, tearoff=False)
            # open
            menu.add_command(label='Open(O)...', under=6, accelerator='Ctrl+O',
                             command=self.open_filedialog)
            WidgetUtils.bind_all(self, 'Control', 'O', self.open_filedialog)
            menu.add_command(label='Save(S)...', under=6, accelerator='Ctrl+S',
                             command=self.save_filedialog)
            WidgetUtils.bind_all(self, 'Control', 'S', self.save_filedialog)
            menu.add_separator()
            # exit
            menu.add_command(label='Exit', under=0, accelerator='Ctrl+Shift+Q',
                             command=self.on_application_exit)
            WidgetUtils.bind_all(self, 'Control-Shift', 'Q', self.on_application_exit)
            return menu

        def crate_image_menu() -> tk.Menu:
            """
            イメージメニュー
            """
            menu = tk.Menu(self, tearoff=False)
            menu.add_checkbutton(label="Show Original Image...", accelerator='Ctrl+1',
                                 command=partial(self.toggle_changed, sender=self.color_image),
                                 variable=self.var_original)
            WidgetUtils.bind_all(self, 'Control-KeyPress', '1',
                                 partial(self.toggle_changed, sender=self.color_image, toggle=True))
            menu.add_checkbutton(label="Show GrayScale Image...", accelerator='Ctrl+2',
                                 command=partial(self.toggle_changed, sender=self.gray_scale_image),
                                 variable=self.var_gray_scale)
            WidgetUtils.bind_all(self, 'Control-KeyPress', '2',
                                 partial(self.toggle_changed, sender=self.gray_scale_image, toggle=True))
            return menu

        menu_bar.add_cascade(menu=crate_file_menu(), label='File')
        menu_bar.add_cascade(menu=crate_image_menu(), label='Image')
        return menu_bar

    def scale_reset(self) ->None:
        """
        パラメータのリセット
        """
        self.controls[CONTROLS.ADAPTIVE].set(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        self.controls[CONTROLS.THRESHOLD_TYPE].set(cv2.THRESH_BINARY)
        self.controls[CONTROLS.BLOCK_SIZE].set(11)
        self.controls[CONTROLS.C].set(2)

    def toggle_changed(self, event=None, sender: ImageWindow = None, toggle: bool = False):
        """
        カラー画像/グレースケール画像を別ウィンドウで表示
        呼び出し元：1,メニューのチェックボックス→チェックボックス側で行うので、トグル処理はしない。
                    2,キーボードショートカット→トグル処理を行う。
                    3,LoadImage→画像の表示更新。
        :param event
        :param sender event sender Widget
        :param toggle True…チェックボックスのトグル処理を行う。Falseはしない。
        :return:
        """
        assert isinstance(sender, ImageWindow), 'toggle_changed:{0}'.format(sender)
        # Menu Visible
        var = sender.var  # type:tk.BooleanVar
        if toggle:
            var.set(not var.get())
        visible = var.get()
        if visible:
            img = None
            if sender.tag == cv2.IMREAD_GRAYSCALE:
                img = self.data.gray_scale
            elif sender.tag == cv2.IMREAD_COLOR:
                img = self.data.color
            sender.update_image(img, sender.tag)

        WidgetUtils.set_visible(sender, visible)

    def on_application_exit(self, event=None):
        """
            アプリケーション終了処理
        """
        sys.exit(0)

    def open_filedialog(self, event=None):
        """
            ファイルを開くダイアログ
        """
        file_path = askopenfilename(self)
        if not file_path:  # isEmpty
            return

        ct()
        self.load_image(ImageData(str(file_path)), True)
        ct()
        self.draw(None)
        ct()

    def save_filedialog(self, event=None):
        """
            ファイルを保存ダイアログ
        """
        # create default file name.
        p = Path(self.data.file_name)
        file_name = '_'.join(map(str, (p.stem, *self.get_params()))) + p.suffix
        IMAGE_FILE_TYPES = [('png (*.png)', ImageFormat['PNG']),
                            ('jpg (*.jpg, *.jpeg)', ImageFormat['JPEG']),
                            ('webp (*.webp)', ImageFormat['WEBP']),
                            ('*', '*.*')]
        file_path = filedialog.asksaveasfilename(parent=self,
                                                 filetypes=IMAGE_FILE_TYPES,
                                                 initialfile=file_name,
                                                 title='名前を付けて保存...')
        if not file_path:  # isEmpty
            return
        ImageData.imwrite(file_path, self.label_image.np)
        LOGGER.info('saved:%s', file_path)

    def get_params(self) -> tuple:
        """
        :return:maxValue, adaptiveMethod, thresholdType, blockSize, C
        """
        return 255, self.controls[CONTROLS.ADAPTIVE].get(), \
            self.controls[CONTROLS.THRESHOLD_TYPE].get(), \
            self.controls[CONTROLS.BLOCK_SIZE].get(), \
            self.controls[CONTROLS.C].get()

    def draw(self, event):
        """
            1,adaptiveThreshold params check
            2,history append
            3,Image Draw
                blocksize range:Odd numbers{3,5,7,9,…} intial:3
            @exsample
                in:0,0  out:NG blocksize of even.
                in:2,0  out:NG blocksize of even.
                in:3,10　out:NG size * size - c < 0
                in:5,25 out:OK
        """
        #print(event)
        params = self.get_params()
        _, _, _, block_size, c = params
        if block_size % 2 == 0:
            return
        if (block_size * block_size - c) < 0:
            return
        try:
            # グレースケール画像を2値化
            ct()
            result = cv2.adaptiveThreshold(self.data.gray_scale, *params)
            insert_str = 'ret = cv2.adaptiveThreshold(src, {0})'.format(', '.join(map(str, params)))
            # 先頭に追加
            self.history.appendleft(insert_str)
            self.listbox.delete(0, tk.END)
            for text in self.history:
                self.listbox.insert(tk.END, text)
            ct()
            WidgetUtils.update_image(self.label_image, result)
            ct()
        except BaseException as ex:
            LOGGER.exception(ex)

    def load_image(self, data: ImageData, redraw: bool = False):
        """
            画像を読み込み、画面に表示
        """
        p = Path(data.file_name)
        LOGGER.info('load file:%s', p.name)
        self.data = data
        self.var_file_name.set(p.name)
        self.var_creation_time.set(datetime.fromtimestamp(p.lstat().st_ctime))
        if redraw:
            # self.draw(None)のエラーチェック条件に一致するとメインウィンドウの画像が更新されない。
            # そのため、こちらで更新する。
            WidgetUtils.update_image(self.label_image, self.data.gray_scale)
            # 画像を変更時にオリジナルとグレースケール画像も更新
            self.toggle_changed(sender=self.color_image)
            self.toggle_changed(sender=self.gray_scale_image)


def parse_args(args: list):
    """
        コマンドライン引数の解析
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=PROGRAM_NAME, description='AdaptiveThreshold Simulator')
    parser.add_argument('input_file', metavar=None, nargs='?')
    parser.add_argument('--version', action='version', version='%(prog)s {0}'.format(__version__))
    return parser.parse_args(args)


def main(entry_point=False):
    """
        Entry Point
        画像イメージを非同期で読み込む
        :param entry_point:True アプリを通常起動、False Pytestより起動
    """
    argv = sys.argv[1:]
    if not entry_point:
        # pytestより起動時
        argv.pop()
        argv.append(r'../images/kodim07.png')

    args = parse_args(argv)
    LOGGER.info('args:%s', args)
    root = tk.Tk()
    WidgetUtils.set_visible(root, False)
    # 起動引数で画像ファイルが渡されなかったから、ファイル選択ダイアログを表示する。
    image_file = args.input_file
    if not image_file:  # isEmpty
        image_file = askopenfilename(root)
        if not image_file:  # isEmpty
            return

    data = ImageData(image_file)
    ct()
    WidgetUtils.set_visible(root, True)
    app = Application(root)
    print('#' *30)
    ct()
    app.load_image(data)
    ct()
    app.pack(expand=True, fill=tk.BOTH)
    ct()
    app.draw(None)

    def finish():
        return ct()

    if entry_point:
        app.after(0, finish)
        app.mainloop()
    else:
        return finish()


if __name__ == "__main__":
    main(True)
