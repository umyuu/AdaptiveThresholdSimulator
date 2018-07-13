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

PROGRAM_NAME = 'AdaptiveThreshold'
__version__ = '0.0.3'
# logging
handler = StreamHandler()
handler.setLevel(DEBUG)
logger = getLogger(PROGRAM_NAME)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class ImageData(object):
    def __init__(self, file_name: str):
        self.__file_name = file_name
        self.__color = ImageData.imread(file_name)
        assert self.color is not None, 'cannot open file as image.'
        self.__gray_scale = cv2.cvtColor(self.color, cv2.COLOR_BGRA2GRAY)
        ids = [id(self.color), id(self.gray_scale)]
        assert len(ids) == len(set(ids)), 'Shallow Copy'

    @staticmethod
    def imread(file_name: str, flags: int=cv2.IMREAD_COLOR):
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
        :param params encode params
        """
        try:
            p = Path(file_name)
            retval, buf = cv2.imencode(p.suffix, image, params)
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
    def color(self):
        return self.__color

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
        widget.bind_all('<{0}-{1}>'.format(modifier, letter.upper()), callback)
        # numeric letter multi assign check.
        if not letter.isdecimal():
            widget.bind_all('<{0}-{1}>'.format(modifier, letter.lower()), callback)

    @staticmethod
    def set_visible(widget: tk.Widget, visible: bool=False) -> None:
        if visible:
            widget.deiconify()
        else:
            widget.withdraw()


class ImageWindow(tk.Toplevel):
    """
    カラー画像とグレースケール画像を表示するために、別ウィンドウとする。
    """
    def __init__(self, master=None, cnf={}, **kw):
        super().__init__(master, cnf, **kw)
        self.protocol('WM_DELETE_WINDOW', partial(WidgetUtils.set_visible, widget=self, visible=False))
        self.__label_image = tk.Label(self)
        self.__label_image.pack()
        self.__tag = None

    def set_image(self, img):
        assert img is not None
        self.__label_image.src = ImageTk.PhotoImage(Image.fromarray(img))
        self.__label_image.configure(image=self.__label_image.src)
    @property
    def tag(self) ->int:
        return self.__tag

    @tag.setter
    def tag(self, value:int):
        self.__tag = value
    """
    Todo
    Labelと画像の関連付を行う。
    Local変数だとGarbageCollectionにより参照が消えて、画像が表示されないことがあるため。
    """



class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        self.master.title('AdaptiveThreshold Simulator Ver:{0}'.format(__version__))
        self.master.update_idletasks()
        self.data = None #オリジナル画像
        self.Component = {}
        self.a_side = tk.Frame(self) # 左側のコンテンツ
        self.main_side = tk.Frame(self) # 右側のコンテンツ
        self.var_file_name = tk.StringVar()
        self.color_image = ImageWindow(self)
        self.color_image.tag = 0
        self.gray_scale_image = ImageWindow(self)
        self.gray_scale_image.tag = 1
        self.history = deque(maxlen=12)
        self.menu_bar = self.create_menubar()
        self.master.configure(menu=self.menu_bar)
        self.create_widgets()
        self.a_side.grid(column=0,row=0, sticky=tk.W)
        self.main_side.grid(column=1, row=0, sticky=tk.W)
        #self.a_side.pack(side=tk.LEFT)
        #self.main_side.pack(side=tk.LEFT, fill=tk.Y)
        #self.a_side.pack(side=tk.LEFT)
        #self.main_side.pack(side=tk.LEFT, fill=tk.Y)

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
            self.var_original = tk.BooleanVar(value=False)
            menu.add_checkbutton(label="Show Original Image...", accelerator='Ctrl+1',
                                 command=partial(self.toggle_changed, sender=self.color_image), variable=self.var_original)
            WidgetUtils.bind_all(self, 'Control-KeyPress', '1', partial(self.toggle_changed, sender=self.color_image, toggle=True))
            self.var_gray_scale = tk.BooleanVar(value=False)
            menu.add_checkbutton(label="Show GrayScale Image...", accelerator='Ctrl+2',
                                 command=partial(self.toggle_changed, sender=self.gray_scale_image), variable=self.var_gray_scale)
            WidgetUtils.bind_all(self, 'Control-KeyPress', '2', partial(self.toggle_changed, sender=self.gray_scale_image, toggle=True))
            return menu

        menu_bar.add_cascade(menu=crate_file_menu(), label='File')
        menu_bar.add_cascade(menu=crate_image_menu(), label='Image')
        return menu_bar

    def toggle_changed(self, event=None, sender:ImageWindow=None, toggle:bool=False):
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
        assert sender, 'toggle_changed:{0}'.format(sender)
        # Menu Visible
        l = [self.var_original, self.var_gray_scale]
        var = l[sender.tag]
        if toggle:
            var.set(not var.get())
        visible = var.get()
        if visible:
            img = None
            if sender.tag == 0:
                img = cv2.cvtColor(self.data.color, cv2.COLOR_BGRA2RGB)
            elif sender.tag == 1:
                img = self.data.gray_scale
            sender.set_image(img)

        WidgetUtils.set_visible(sender, visible)

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
        # create default file name.
        p = Path(self.data.file_name)
        file_name =  '_'.join(map(str, (p.stem, *self.get_params()))) + p.suffix
        IMAGE_FILE_TYPES = [('png (*.png)', '*.png'), ('jpg (*.jpg, *.jpeg)', ("*.jpg", "*.jpeg")), ('*', '*.*')]
        file_path = filedialog.asksaveasfilename(parent=self,
                                                 filetypes=IMAGE_FILE_TYPES,
                                                 initialfile=file_name,
                                                 title='名前を付けて保存...')
        if len(file_path) == 0:
            return
        ImageData.imwrite(file_path, self.label_image.np)
        logger.info('saved:{0}'.format(file_path))

    def params_frame(self):
        controls = dict()
        self.top_frame = tk.LabelFrame(self.a_side, text='params')
        self.top_frame.pack(side=tk.TOP)

        controls['ADAPTIVE'] = {'label': '0:MEAN_C / 1:GAUSSIAN_C',
                                'from_': cv2.ADAPTIVE_THRESH_MEAN_C, 'to': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_adaptive = tk.Scale(self.top_frame, controls['ADAPTIVE'])
        self.scale_adaptive.pack()

        controls['THRESHOLDTYPE'] = {'label': '0:BINARY / 1:INV',
                                     'from_': cv2.THRESH_BINARY, 'to': cv2.THRESH_BINARY_INV,
                                     'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_thresholdType = tk.Scale(self.top_frame, controls['THRESHOLDTYPE'])
        self.scale_thresholdType.pack()
        # initial stepvalue 3.
        controls['BLOCKSIZE'] = {'label': 'blocksize', 'from_': 3, 'to': 255,
                                 'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_blocksize = tk.Scale(self.top_frame, controls['BLOCKSIZE'])
        self.scale_blocksize.pack()
        controls['C'] = {'label': 'c', 'from_': 0, 'to': 255,
                         'length': 300, 'orient': tk.HORIZONTAL, 'command': self.draw}
        self.scale_c = tk.Scale(self.top_frame, controls['C'])
        self.scale_c.pack()

        self.scale_reset()

    def command_frame(self):
        self.command_frame = tk.Frame(self.a_side)
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
        self.output_frame = tk.LabelFrame(self.a_side, text='output')
        self.output_frame.pack(side=tk.TOP, fill=tk.Y)

        #self.label_message = tk.Message(self.output_frame, text='Select a row and Ctrl+C\nCopy it to the clipboard.', width=200)
        self.label_message = tk.Label(self.output_frame, text='Select a row and Ctrl+C\nCopy it to the clipboard.')
        self.label_message.pack(expand=True,side=tk.TOP,fill=tk.X)

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

        self.message_panel = tk.Label(self.main_side, text='Ctrl+S…Image Save Dialog')
        self.message_panel.pack(side=tk.TOP, fill=tk.Y)
        self.entry_filename = tk.Entry(self.main_side, textvariable=self.var_file_name)
        self.entry_filename.pack(side=tk.TOP, fill=tk.Y)
        self.label_image = tk.Label(self.main_side)
        self.label_image.pack(side=tk.TOP, fill=tk.Y)

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
        self.var_file_name.set(p.name)
        self.__change_image(self.data.gray_scale)
        # 画像を変更時にオリジナルとグレースケール画像も更新
        self.toggle_changed(sender=self.color_image)
        self.toggle_changed(sender=self.gray_scale_image)

    def __change_image(self, src):
        self.label_image.np = src
        self.label_image.src = ImageTk.PhotoImage(Image.fromarray(src))
        self.label_image.configure(image=self.label_image.src)


def main():
    input_file = r'../images/kodim07.png'
    #input_file = r'../images/桜_768-512.jpg'
    parser = ArgumentParser(prog=PROGRAM_NAME, description='AdaptiveThreshold Simulator')
    parser.add_argument('input_file', metavar=None, nargs='?', default=input_file)
    parser.add_argument('--version', action='version', version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args()
    logger.info('args:{0}'.format(args))
    
    app = Application()
    app.load_image(args.input_file)
    app.pack()
    app.mainloop()


if __name__ == "__main__":
    main()
