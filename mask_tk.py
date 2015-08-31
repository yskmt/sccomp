"""
create_mask.py

Interactively create a mask from a source image.

Requirements:
* Tkinter
* PIL

"""

import Tkinter
import tkMessageBox
from PIL import Image, ImageTk, ImageDraw


class callback_obj:
    def __init__(self, radius, draw, top, mask_image, mask_name):
        self.radius = radius
        self.draw = draw
        self.top = top
        self.mask_image = mask_image
        self.mask_name = mask_name

    def mouse_down(self, event):
        w = event.widget
        x, y = event.x, event.y

        w.create_oval(x - self.radius, y - self.radius,
                      x + self.radius, y + self.radius,
                      fill="blue", outline="blue")

        self.draw.ellipse(
            [x - self.radius, y - self.radius,
             x + self.radius, y + self.radius], fill="white")

        print "mouse position at", x, y

    def on_closing(self):
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            self.mask_image.save(self.mask_name)
            self.top.destroy()
        

def create_mask(img_name, mask_name, radius=10):

    # open image by PIL
    image = Image.open(img_name)
    wi, hi = image.size

    # create a widget
    top = Tkinter.Tk()
    w = Tkinter.Canvas(top, height=hi, width=wi)
    w.pack()
    # create a mask image by PIL
    mask_image = Image.new("RGB", (wi, hi), color="black")
    draw = ImageDraw.Draw(mask_image)

    # draw an image
    photo = ImageTk.PhotoImage(image)
    image = w.create_image(0, 0, anchor=Tkinter.NW, image=photo)
    
    # bind event
    cbo = callback_obj(radius, draw, top, mask_image, mask_name)
    w.bind("<B1-Motion>", cbo.mouse_down)
    w.bind("<ButtonPress-1>", cbo.mouse_down)
    top.protocol("WM_DELETE_WINDOW", cbo.on_closing)

    # show
    top.mainloop()
    
# create_mask('data/scene_database/coast/n238045.jpg', 'mask.png')
