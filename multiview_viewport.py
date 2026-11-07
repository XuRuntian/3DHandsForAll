"""Aspect-correct Tk image viewport: fit/cover, cursor zoom and drag to pan."""
import tkinter as tk
from PIL import Image, ImageTk


def image_geometry(image_size, viewport_size, mode='fit', zoom=1., pan=(0., 0.)):
    iw, ih = image_size
    vw, vh = viewport_size
    ratio = (max if mode == 'cover' else min)(vw / iw, vh / ih) * zoom
    width, height = max(1, round(iw * ratio)), max(1, round(ih * ratio))
    return width, height, (vw-width)/2 + pan[0], (vh-height)/2 + pan[1]


class ImageViewport(tk.Canvas):
    def __init__(self, parent, on_select=None, on_focus=None):
        super().__init__(parent, bg='#111923', highlightthickness=0, width=1, height=1,
                         cursor='hand2', bd=0)
        self.source = None
        self.photo = None
        self.mode = 'fit'
        self.zoom = 1.
        self.pan = (0., 0.)
        self.pending = None
        self.drag = None
        self.item = self.create_image(0, 0, anchor='nw')
        self.bind('<Configure>', lambda e: self.schedule())
        self.bind('<ButtonPress-1>', lambda e: self.start_drag(e, on_select))
        self.bind('<B1-Motion>', self.move_drag)
        self.bind('<Double-Button-1>', lambda e: on_focus() if on_focus else self.reset())
        self.bind('<MouseWheel>', lambda e: self.zoom_at(1.15 if e.delta > 0 else 1/1.15, e.x, e.y))
        self.bind('<Button-4>', lambda e: self.zoom_at(1.15, e.x, e.y))
        self.bind('<Button-5>', lambda e: self.zoom_at(1/1.15, e.x, e.y))
        self.bind('<Destroy>', self.cleanup, add='+')

    def cleanup(self, event):
        if event.widget == self and self.pending is not None:
            self.after_cancel(self.pending)
            self.pending = None

    def set_image(self, rgb):
        self.source = Image.fromarray(rgb)
        self.draw()

    def set_mode(self, mode):
        self.mode = mode
        self.reset()

    def reset(self):
        self.zoom, self.pan = 1., (0., 0.)
        self.draw()

    def schedule(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(25, self.draw)

    def start_drag(self, event, on_select):
        self.focus_set()
        if on_select:
            on_select()
        self.drag = (event.x, event.y, self.pan)

    def move_drag(self, event):
        if self.drag:
            x, y, pan = self.drag
            self.pan = (pan[0]+event.x-x, pan[1]+event.y-y)
            self.schedule()

    def zoom_at(self, factor, x=None, y=None):
        new_zoom = max(1., min(8., self.zoom * factor))
        factor = new_zoom / self.zoom
        cx, cy = self.winfo_width()/2, self.winfo_height()/2
        x, y = (cx if x is None else x), (cy if y is None else y)
        self.pan = (factor*self.pan[0] + (1-factor)*(x-cx),
                    factor*self.pan[1] + (1-factor)*(y-cy))
        self.zoom = new_zoom
        if self.zoom == 1.:
            self.pan = (0., 0.)
        self.draw()

    def draw(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
            self.pending = None
        if self.source is None or not self.winfo_exists():
            return
        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        if vw <= 1 or vh <= 1:
            return
        w, h, x, y = image_geometry(self.source.size, (vw, vh), self.mode, self.zoom, self.pan)
        # Resample only the visible rectangle; 8x zoom never allocates an 8x RGB.
        sx, sy = self.source.width / w, self.source.height / h
        self.photo = ImageTk.PhotoImage(self.source.transform(
            (vw, vh), Image.Transform.AFFINE, (sx, 0, -x*sx, 0, sy, -y*sy),
            resample=Image.Resampling.BILINEAR, fillcolor=(17,25,35)))
        self.itemconfigure(self.item, image=self.photo)
        self.coords(self.item, 0, 0)
