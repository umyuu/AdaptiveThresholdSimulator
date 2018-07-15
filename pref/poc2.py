# -*- coding: utf-8 -*-
"""
    poc 2
    opencvを使って、PhotoImageの処理を代替する。
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
#from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from contextlib import closing
from functools import partial
from logging import getLogger, DEBUG, StreamHandler
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
# library
import cv2
from numpy import asarray, uint8

# logging
HANDLER = StreamHandler()
HANDLER.setLevel(DEBUG)
LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
LOGGER.addHandler(HANDLER)

#LOOP = asyncio.get_event_loop()
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)
LOOP.set_default_executor(PoolExecutor(8))


def read_file(file_name: str):
    p = Path(file_name)
    with p.open('rb') as file:
        return file.read()


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
        MSG = [func_name, n, end_time, (end_time - start_time)]
        LOGGER.debug(MSG)
        return MSG, end_time - start_time

    return step

ct = stop_watch()

class WidgetUtils(object):
    @staticmethod
    def bind_all(widget: tk.Widget, modifier: str = "", letter: str = "", callback=None) -> None:
        widget.bind_all('<{0}-{1}>'.format(modifier, letter.upper()), callback)
        # numeric letter multi assign check.
        if not letter.isdecimal():
            widget.bind_all('<{0}-{1}>'.format(modifier, letter.lower()), callback)


class ImageUtils(object):
    @staticmethod
    async def imread(file_name: str, flags: int = cv2.IMREAD_COLOR):
        image = None
        try:
            val = await LOOP.run_in_executor(None, partial(read_file, file_name))
            buffer = asarray(bytearray(val), dtype=uint8)
            image = cv2.imdecode(buffer, flags)
        except FileNotFoundError as ex:
            # cv2.imread compatible
            LOGGER.exception(ex)
        return image

    @staticmethod
    def imencode(image, flags: int):
        """
        読み込んだ画像データをPGMまたはPPM形式にエンコードする。
        :param image:
        :param flags:
        :return:
        """
        table = {cv2.IMREAD_GRAYSCALE: '.PGM', cv2.IMREAD_COLOR: '.PPM'}
        ext = table.get(flags)
        assert ext
        result, enc_img = cv2.imencode(ext, image, None)
        assert result, 'imencode failure'
        return enc_img.tobytes()


class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        self.master.geometry("1280x1024")
        self.master.title("PoC 2")
        self.master.update_idletasks()
        #
        self.file_path = None
        # メニューの作成
        self.menu_bar = self.create_menubar()
        self.master.configure(menu=self.menu_bar)
        # コンテンツ領域
        self.a_side = tk.Frame(self) # 左側のコンテンツ
        self.main_side = tk.Frame(self) # 右側のコンテンツ

        self.create_params_frame()
        self.label_image = tk.Label(self.main_side)
        self.label_image.pack(anchor=tk.NW, pady=10)
        self.a_side.pack(side=tk.LEFT, anchor=tk.NW)
        self.main_side.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, anchor=tk.NW)

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
            return menu

        menu_bar.add_cascade(menu=crate_file_menu(), label='File')
        return menu_bar

    def create_params_frame(self):
        self.params_frame = tk.LabelFrame(self.a_side, text='params')
        self.params_frame.pack()
        controls = dict()
        controls['READ_MODE'] = {'label': '0:GRAY_SCALE / 1:COLOR',
                                 'from_': cv2.IMREAD_GRAYSCALE, 'to': cv2.IMREAD_COLOR,
                                 'length': 300, 'orient': tk.HORIZONTAL}
        self.scale_read_mode = tk.Scale(self.params_frame, controls['READ_MODE'])
        self.scale_read_mode.pack()
        controls['ENCODE_METHOD'] = {'label': '0:Opencv3 / 1:Pillow',
                                 'from_': 0, 'to': 1,
                                 'length': 300, 'orient': tk.HORIZONTAL}
        self.scale_encode_method = tk.Scale(self.params_frame, controls['ENCODE_METHOD'])
        self.scale_encode_method.pack()
        #
        self.scale_read_mode.configure(command=self.draw)
        self.scale_encode_method.configure(command=self.draw)

    def set_image_opencv(self, img):
        ct()
        self.label_image.src = tk.PhotoImage(data=ImageUtils.imencode(img, self.scale_read_mode.get()))
        self.label_image.configure(image=self.label_image.src)
        ct()

    def set_image_pillow(self, img):
        from PIL import Image, ImageTk
        ct()
        dst = None
        read_mode = self.scale_read_mode.get()
        if read_mode == cv2.IMREAD_COLOR:
            dst = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif read_mode == cv2.IMREAD_GRAYSCALE:
            dst = img
        else:
            assert dst
        self.label_image.src = ImageTk.PhotoImage(Image.fromarray(dst))
        self.label_image.configure(image=self.label_image.src)
        ct()



    def open_filedialog(self, event=None):
        """
        ファイルを開くダイアログ
        :param event:
        :return:
        """
        file_path = filedialog.askopenfilename(parent=self)
        if not file_path:  # isEmpty
            return
        self.file_path = file_path
        self.after(0, self.draw, (None, ))

    def draw(self, event):
        if not self.file_path:  # isEmpty
            return

        ct()
        val = LOOP.run_until_complete(LOOP.create_task(ImageUtils.imread(self.file_path, self.scale_read_mode.get())))
        ct()
        enc_method = self.scale_encode_method.get()
        if  enc_method == 0:
            self.set_image_opencv(val)
        elif enc_method == 1:
            self.set_image_pillow(val)
        else:
            assert False, 'enc_method'
        ct()


def main():
    """
        Entry Point
    """
    app = Application()
    app.pack()
    app.mainloop()


if __name__ == "__main__":
    main()
