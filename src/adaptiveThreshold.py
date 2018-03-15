# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from functools import partial
from logging import getLogger, DEBUG, StreamHandler
from pathlib import Path
import sys

import tkinter as tk
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
    def __init__(self, master=None):
        super().__init__(master)
        self.master.title('AdaptiveThreshold Simulator')
        self.data = None
        self.photo_image = None
        self.menu_bar = self.create_menu()
        self.master.configure(menu=self.menu_bar)
        self.create_widgets()

    def create_menu(self) -> tk.Menu:
        menu_bar = tk.Menu(self)
        menu_file = tk.Menu(self)
        # open
        menu_file.add_command(label='Open(O)...', under=0, accelerator='Ctrl+O',
                              command=partial(self.open_filedialog, event=None))
        self.bind_all('<Control-O>', self.open_filedialog)
        self.bind_all('<Control-o>', self.open_filedialog)
        # exit
        menu_file.add_command(label='Exit', under=0, accelerator='Ctrl+Shift+Q',
                              command=partial(self.on_application_exit, event=None))
        self.bind_all('<Control-Shift-Q>', self.on_application_exit)
        self.bind_all('<Control-Shift-q>', self.on_application_exit)

        menu_bar.add_cascade(menu=menu_file, label='File')
        return menu_bar

    def on_application_exit(self, event):
        sys.exit(0)

    def open_filedialog(self, event):
        file_path = filedialog.askopenfilename()
        if len(file_path) == 0:
            return
        self.load_image(file_path)

    def create_widgets(self):
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
        self.scale_c.set(2)
        self.scale_c.pack()
        
        self.lblimage = tk.Label(self)
        self.lblimage.grid(row=1, column=0)

    def draw(self, event):
        size = self.scale_blocksize.get()
        c = self.scale_c.get()
        # adaptiveThreshold params check
        # blocksize range:Odd numbers{3,5,7,9,…} intial:3
        #   in:0,0  out:NG blocksize of even.
        #   in:2,0  out:NG blocksize of even.
        #   in:3,10　out:NG size * size - c < 0
        #   in:5,25 out:OK
        if size % 2 == 0:
            return
        if (size * size - c) < 0:
            return
        try:
            result = cv2.adaptiveThreshold(self.data.gray_scale, 255,
                                           self.scale_adaptive.get(), self.scale_thresholdType.get(), size, c)
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
