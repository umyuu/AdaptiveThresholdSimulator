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
    def __init__(self, file_name: str):
        self.__file_name = file_name
        self.__src = ImageData.imread(file_name)
        assert self.src is not None, 'image file empty'
        self.__canvas = self.src.copy()
        self.__gray_scale = cv2.cvtColor(self.canvas, cv2.COLOR_BGRA2GRAY)
        ids = [id(self.src), id(self.canvas), id(self.gray_scale)]
        assert len(ids) == len(set(ids)), 'shallow copy'

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
        :return: {Mat}image BGR
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

    @staticmethod
    def imwrite(file_name: str, image, params=None) -> bool:
        """
        Unicode Path/Filename for imwrite Not supported.
        cv2.imwrite alternative = cv2.imencode & numpy.ndarray.tofile
        :param file_name
        :param image imagedata
        :param params encode_param
        """
        try:
            p = Path(file_name)
            retval, buf  = cv2.imencode(p.suffix, image, params)
            if not retval:
                return retval
            with p.open('wb') as f:
                buf.tofile(f)
            return True
        except IOError as ex:
            logger.error(ex)
            pass
        return False

    @property
    def file_name(self) -> str:
        return self.__file_name

    @property
    def src(self):
        return self.__src

    @property
    def canvas(self):
        return self.__canvas

    @property
    def gray_scale(self):
        return self.__gray_scale


class WidgetUtils(object):
    @staticmethod
    def bind_all(widget: tk.Widget, modifier: str="", letter: str="", callback=None) -> None:
        """
        Keyboard Shortcut Assign.
        :param widget:
        :param modifier:
        :param letter:
        :param callback:
        :return:
        """
        # numelic letter multi assign check.
        upper = letter.upper()
        lower = letter.lower()
        widget.bind_all('<{0}-{1}>'.format(modifier, upper), callback)
        if not upper == lower:
            widget.bind_all('<{0}-{1}>'.format(modifier, lower), callback)

    @staticmethod
    def set_visible(widget: tk.Widget, visible: bool=False) -> None:
        if visible:
            widget.deiconify()
        else:
            widget.withdraw()


class ImageLabel(tk.Label):
    """
    Labelと画像の関連付を行う。
    Local変数だとGarbageCollectionにより参照が消えて、
    """
    def __init__(self, master=None, cnf={}, **kw):
        super().__init__(master, cnf, **kw)
        self.__src = None

    @property
    def src(self):
        return self.__src


class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        self.master.title('AdaptiveThreshold Simulator')
        self.master.update_idletasks()
        self.data = None #オリジナル画像
        self.photo_image = None
        self.Component = {}
        self.aside = tk.Frame(self)
        self.src_image = tk.Toplevel(self)
        self.src_image.protocol('WM_DELETE_WINDOW',
                                partial(WidgetUtils.set_visible, widget=self.src_image, visible=False))
        WidgetUtils.set_visible(self.src_image, False)
        self.gray_scale_image = tk.Toplevel(self)
        self.gray_scale_image.protocol('WM_DELETE_WINDOW',
                                       partial(WidgetUtils.set_visible, widget=self.gray_scale_image, visible=False))
        WidgetUtils.set_visible(self.gray_scale_image, False)

        self.history = deque(maxlen=12)
        self.menu_bar = self.create_menubar()
        self.master.configure(menu=self.menu_bar)
        self.create_widgets()

        self.aside.grid(row=0, column=0)

    def create_menubar(self) -> tk.Menu:
        """
        MenuBarの作成
        :return:
        """
        menu_bar = tk.Menu(self, tearoff=False)

        def crate_file_menu() -> tk.Menu:
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
            menu = tk.Menu(self, tearoff=False)
            self.var_original = tk.BooleanVar()
            self.var_original.set(False)
            menu.add_checkbutton(label="Show Original Image...", accelerator='Ctrl+A',
                                 command=partial(self.toggle_changed, param=1), variable=self.var_original)
            WidgetUtils.bind_all(self, 'Control', 'A', partial(self.toggle_changed, param=1))
            self.var_gray_scale = tk.BooleanVar()
            self.var_gray_scale.set(False)
            menu.add_checkbutton(label="Show GrayScale Image...", accelerator='Ctrl+B',
                                 command=partial(self.toggle_changed, param=2), variable=self.var_gray_scale)
            WidgetUtils.bind_all(self, 'Control', 'B', partial(self.toggle_changed, param=2))
            return menu

        menu_bar.add_cascade(menu=crate_file_menu(), label='File')
        menu_bar.add_cascade(menu=crate_image_menu(), label='Image')
        return menu_bar

    def toggle_changed(self, event=None, param=0):
        """
        TopLevel Windowを表示
        :param event
        :param param: event sender      1:Src Image, 2:GrayScale Image
        :return:
        """
        if param == 1:
            parent = self.lblimage_original.master
            visible = self.var_original.get()
            if visible:
                rgb = cv2.cvtColor(self.data.canvas, cv2.COLOR_BGRA2RGB)
                self.lblimage_original.src = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.lblimage_original.configure(image=self.lblimage_original.src)
            WidgetUtils.set_visible(parent, visible)
            return
        elif param == 2:
            parent = self.lblimage_gray_scale.master
            visible = self.var_gray_scale.get()
            if visible:
                self.lblimage_gray_scale.src = ImageTk.PhotoImage(Image.fromarray(self.data.gray_scale))
                self.lblimage_gray_scale.configure(image=self.lblimage_gray_scale.src)
            WidgetUtils.set_visible(parent, visible)
            return
        assert False, 'toggle_changed:{0}'.format(param)

    def on_application_exit(self, event=None):
        sys.exit(0)

    def open_filedialog(self, event=None):
        ALL_IMAGE = ('Image Files', ('*.png', '*.jpg', '*.jpeg'))
        IMAGE_FILE_TYPES = [ALL_IMAGE, ('png (*.png)', '*.png'),
                            ('jpg (*.jpg, *.jpeg)', ("*.jpg", "*.jpeg")), ('*', '*.*')]
        file_path = filedialog.askopenfilename(parent=self,
                                               filetypes=IMAGE_FILE_TYPES)
        if len(file_path) == 0:
            return
        self.load_image(file_path)
        self.draw(None)

    def save_filedialog(self, event=None):
        # create defalut file name.
        p = Path(self.data.file_name)
        file_name =  '_'.join(map(str, (p.stem, *self.get_params()))) + p.suffix
        IMAGE_FILE_TYPES = [('png (*.png)', '*.png'), ('jpg (*.jpg, *.jpeg)', ("*.jpg", "*.jpeg")), ('*', '*.*')]
        file_path = filedialog.asksaveasfilename(parent=self,
                                                 filetypes=IMAGE_FILE_TYPES,
                                                 initialfile=file_name,
                                                 title='名前を付けて保存...')
        if len(file_path) == 0:
            return

        ImageData.imwrite(file_path, self.lblimage.np)
        logger.info('saved:{0}'.format(file_path))

    def params_frame(self):
        controls = dict()
        self.topframe = tk.LabelFrame(self.aside, text='params')

        self.topframe.pack(side=tk.TOP)
        #self.topframe.grid(row=0, column=0)

        controls['ADAPTIVE'] = {'label': '0:MEAN_C / 1:GAUSSIAN_C',
                                'from_': cv2.ADAPTIVE_THRESH_MEAN_C, 'to': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_adaptive = tk.Scale(self.topframe, controls['ADAPTIVE'])
        self.scale_adaptive.pack()

        controls['THRESHOLDTYPE'] = {'label': '0:BINARY / 1:INV',
                                     'from_': cv2.THRESH_BINARY, 'to': cv2.THRESH_BINARY_INV,
                                     'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_thresholdType = tk.Scale(self.topframe, controls['THRESHOLDTYPE'])
        self.scale_thresholdType.pack()
        # initial stepvalue 3.
        controls['BLOCKSIZE'] = {'label': 'blocksize', 'from_': 3, 'to': 255,
                                 'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_blocksize = tk.Scale(self.topframe, controls['BLOCKSIZE'])
        self.scale_blocksize.pack()
        controls['C'] = {'label': 'c', 'from_': 0, 'to': 255,
                         'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_c = tk.Scale(self.topframe, controls['C'])

        self.scale_c.pack()
        self.scale_reset()

    def command_frame(self):
        self.command_frame = tk.Frame(self.aside)
        self.button_reset = tk.Button(self.command_frame, text='RESET', command=self.scale_reset)
        self.button_reset.pack()
        self.command_frame.pack()

    def scale_reset(self):
        self.scale_adaptive.set(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        self.scale_thresholdType.set(cv2.THRESH_BINARY)
        self.scale_blocksize.set(11)
        self.scale_c.set(2)

    def get_params(self) -> tuple:
        """
        :return:maxValue, adaptiveMethod, thresholdType, blockSize, C
        """
        return 255, self.scale_adaptive.get(), self.scale_thresholdType.get(), self.scale_blocksize.get(), self.scale_c.get()

    def output_frame(self):
        self.output_frame = tk.LabelFrame(self.aside, text='output')
        #self.output_frame.grid(row=0, column=1)
        #self.output_frame.grid(row=1, column=0)
        self.output_frame.pack(side=tk.TOP, fill=tk.Y)
        self.message = tk.Label(self.output_frame, text='Select a row and CTRL+C\nCopy it to the clipboard.')
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
        self.command_frame()
        self.output_frame()

        self.lblimage_original = tk.Label(self.src_image)
        self.lblimage_original.pack()
        self.lblimage_gray_scale = tk.Label(self.gray_scale_image)
        self.lblimage_gray_scale.pack()
        self.lblimage = tk.Label(self)
        self.lblimage.grid(row=0, column=1)
        #self.lblimage.grid(row=1, columnspan=2)

    def draw(self, event):
        #print(event)
        params = self.get_params()
        max_value, adaptive_method, threshold_type, block_size, c = params
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
            result = cv2.adaptiveThreshold(self.data.gray_scale, *params)
            insert_str = 'ret = cv2.adaptiveThreshold(src, {0})'.format(', '.join(map(str, params)))
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
        self.data = ImageData(str(p))
        self.__change_image(self.data.gray_scale)
        # 画像を変更時にオリジナルとグレースケール画像も更新
        self.toggle_changed(param=1)
        self.toggle_changed(param=2)

    def __change_image(self, src):
        self.lblimage.np = src
        self.lblimage.src = ImageTk.PhotoImage(Image.fromarray(src))
        self.lblimage.configure(image=self.lblimage.src)


def main():
    input_file = r'../images/kodim07.png'
    #input_file = r'../images/桜_768-512.jpg'
    parser = ArgumentParser(prog=PROGRAM_NAME, description='AdaptiveThreshold Simulator')
    parser.add_argument('input_file', metavar=None, nargs='?', default=input_file)
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.3')
    args = parser.parse_args()
    logger.info('args:{0}'.format(args))
    
    app = Application()
    app.load_image(args.input_file)
    app.pack()
    app.mainloop()


if __name__ == "__main__":
    main()
