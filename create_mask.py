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

radius = 10


def callback(event):
    w = event.widget
    x, y = event.x, event.y

    w.create_oval(x - radius, y - radius, x + radius, y + radius,
                  fill="blue", outline="blue")

    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], fill="white")

    print "mouse position at", x, y


def on_closing():
    if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
        top.destroy()

        filename = "mask.png"
        mask_image.save(filename)

# open image by PIL
image = Image.open('testimages/test1_src.png')
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
w.bind("<B1-Motion>", callback)
w.bind("<ButtonPress-1>", callback)
top.protocol("WM_DELETE_WINDOW", on_closing)

# show
top.mainloop()
