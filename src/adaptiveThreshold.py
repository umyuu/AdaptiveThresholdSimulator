# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from collections import deque
from functools import partial
from logging import getLogger, DEBUG, StreamHandler
from pathlib import Path
import sys

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
# library
import numpy as np
import cv2
from PIL import Image, ImageTk

PROGRAM_NAME = 'adaptiveThreshold'
# logging
handler = StreamHandler()
handler.setLevel(DEBUG)
logger = getLogger(PROGRAM_NAME)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class ImageData(object):
    def __init__(self, src):
        assert src is not None
        self.__canvas = src.copy()
        self.__gray_scale = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def imread(file_name: str, flags: int=cv2.IMREAD_COLOR):
        """
        Unicode Path/Filename for imread Not supported.
        ■ref
            https://github.com/opencv/opencv/issues/4292

        cv2.imread alternative = np.asarray & cv2.imdecode
        Unicode Path/Filename image file read.
        :param file_name:
        :param flags: cv2.IMREAD_COLOR
        :return: {Mat}image
            FileNotFoundError image is None
        """
        image = None
        try:
            with open(file_name, 'rb') as file:
                buffer = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(buffer, flags)
        except FileNotFoundError as ex:
            # cv2.imread compatible
            logger.error(ex)
            pass
        return image

    @property
    def canvas(self):
        return self.__canvas

    @property
    def gray_scale(self):
        return self.__gray_scale


class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        self.master.title('AdaptiveThreshold Simulator')
        self.master.update_idletasks()
        self.data = None
        self.photo_image = None
        self.Component = {}
        self.history = deque(maxlen=12)
        self.menu_bar = self.create_menu()
        self.master.configure(menu=self.menu_bar)
        self.create_widgets()

    def create_menu(self) -> tk.Menu:
        menu_bar = tk.Menu(self, tearoff=False)

        def crate_file_menu() -> tk.Menu:
            menu = tk.Menu(self, tearoff=False)
            # open
            menu.add_command(label='Open(O)...', under=6, accelerator='Ctrl+O',
                             command=partial(self.open_filedialog, event=None))
            self.bind_all('<Control-O>', self.open_filedialog)
            self.bind_all('<Control-o>', self.open_filedialog)
            menu.add_separator()
            # exit
            menu.add_command(label='Exit', under=0, accelerator='Ctrl+Shift+Q',
                             command=partial(self.on_application_exit, event=None))
            self.bind_all('<Control-Shift-Q>', self.on_application_exit)
            self.bind_all('<Control-Shift-q>', self.on_application_exit)
            return menu

        def crate_image_menu() -> tk.Menu:
            menu = tk.Menu(self, tearoff=False)
            menu.add_command(label='Src Image(S)...', under=6, accelerator='Ctrl+S',
                             command=partial(self.open_filedialog, event=None))
            return menu

        menu_bar.add_cascade(menu=crate_file_menu(), label='File')
        menu_bar.add_cascade(menu=crate_image_menu(), label='Image')
        return menu_bar

    def on_application_exit(self, event):
        sys.exit(0)

    def open_filedialog(self, event):
        file_path = filedialog.askopenfilename()
        if len(file_path) == 0:
            return
        self.load_image(file_path)
        self.draw(None)

    def params_frame(self):
        controls = dict()
        self.topframe = tk.LabelFrame(self, text='params')
        self.topframe.grid(row=0, column=0)

        controls['ADAPTIVE'] = {'label': '0:MEAN_C / 1:GAUSSIAN_C',
                                'from_': cv2.ADAPTIVE_THRESH_MEAN_C, 'to': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_adaptive = tk.Scale(self.topframe, controls['ADAPTIVE'])
        self.scale_adaptive.set(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        self.scale_adaptive.pack()

        controls['THRESHOLDTYPE'] = {'label': '0:BINARY / 1:INV',
                                     'from_': cv2.THRESH_BINARY, 'to': cv2.THRESH_BINARY_INV,
                                     'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_thresholdType = tk.Scale(self.topframe, controls['THRESHOLDTYPE'])
        self.scale_thresholdType.set(cv2.THRESH_BINARY)
        self.scale_thresholdType.pack()
        # initial stepvalue 3.
        controls['BLOCKSIZE'] = {'label': 'blocksize', 'from_': 3, 'to': 255,
                                 'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_blocksize = tk.Scale(self.topframe, controls['BLOCKSIZE'])
        self.scale_blocksize.set(11)
        self.scale_blocksize.pack()

        controls['C'] = {'label': 'c', 'from_': 0, 'to': 255,
                         'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_c = tk.Scale(self.topframe, controls['C'])

        self.scale_c.pack()
        self.scale_c.set(2)

    def get_params(self) -> tuple:
        """
        :return:maxValue, adaptiveMethod, thresholdType, blockSize, C
        """
        return 255, self.scale_adaptive.get(), self.scale_thresholdType.get(), self.scale_blocksize.get(), self.scale_c.get()

    def output_frame(self):
        self.output_frame = tk.LabelFrame(self, text='output')
        self.output_frame.grid(row=0, column=1)
        self.message = tk.Label(self.output_frame, text='Select a row and CTRL+C: Copy it to the clipboard.')
        self.message.pack()

        class ScrollListBox(tk.Listbox):
            def __init__(self, master=None, cnf={}, **kw):
                super().__init__(master, cnf, **kw)
                self.y_scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL, command=self.yview)
                self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.configure(yscrollcommand=self.y_scrollbar.set)
                self.pack(side=tk.LEFT, fill=tk.Y)

        self.listbox = ScrollListBox(self.output_frame, width=40, height=self.history.maxlen)

    def create_widgets(self):
        self.params_frame()
        self.output_frame()
        self.lblimage = tk.Label(self)
        self.lblimage.grid(row=1, columnspan=2)

    def draw(self, event):
        max_value, adaptive_method, threshold_type, block_size, c = self.get_params()
        # adaptiveThreshold params check
        # blocksize range:Odd numbers{3,5,7,9,…} intial:3
        #   in:0,0  out:NG blocksize of even.
        #   in:2,0  out:NG blocksize of even.
        #   in:3,10　out:NG size * size - c < 0
        #   in:5,25 out:OK
        if block_size % 2 == 0:
            return
        if (block_size * block_size - c) < 0:
            return
        try:
            result = cv2.adaptiveThreshold(self.data.gray_scale, max_value,
                                           adaptive_method, threshold_type, block_size, c)
            insert_str = 'cv2.adaptiveThreshold(src, {0})'.format(', '.join(map(str, self.get_params())))
            # 先頭に追加
            self.history.appendleft(insert_str)
            self.listbox.delete(0, tk.END)
            for text in self.history:
                self.listbox.insert(tk.END, text)

            self.__change_image(result)
        except Exception as ex:
            print(ex)
            pass

    def load_image(self, file_path: str):
        p = Path(file_path)
        logger.info('load file:{0}'.format(p.name))
        src = ImageData.imread(str(p))
        self.data = ImageData(src)
        self.__change_image(src)

    def __change_image(self, src):
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(src))
        self.lblimage.configure(image=self.photo_image)


def main():
    input_file = r'../images/kodim07.png'
    #input_file = r'../images/桜_768-512.jpg'
    parser = ArgumentParser(prog=PROGRAM_NAME, description='AdaptiveThreshold Simulator')
    parser.add_argument('input_file', metavar=None, nargs='?', default=input_file)
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.2')
    args = parser.parse_args()
    logger.info('args:{0}'.format(args))
    
    app = Application()
    app.load_image(args.input_file)
    app.pack()
    app.mainloop()


if __name__ == "__main__":
    main()
