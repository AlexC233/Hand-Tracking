import tkinter as tk
from PIL import ImageTk, Image

# SETUP.
root = tk.Tk()
WIDTH = 1980
HEIGHT = 1080
# root.maxsize(1920, 1820)

# Errors.
flat_fingers = {
    "text": "Lift Your Fingers!",
    "image": ImageTk.PhotoImage(Image.open("./images/Wrist_TopLeft.png").resize(((WIDTH//4, HEIGHT//4))))
}

raise_wrists = {
    "text": "Raise Your Wrists!",
    "image": ImageTk.PhotoImage(Image.open("./images/Fingers_TopRight.png").resize((WIDTH//4, HEIGHT//4)))
}

flying_pinkie = {
    "text": "Bend Your Pinky!",
    "image": ImageTk.PhotoImage(Image.open("./images/Thumb_BottomLeft.png").resize((WIDTH//4, HEIGHT//4)))
}

thumb_falling = {
    "text": "Put Your Thumb on the Keyboard!",
    "image": ImageTk.PhotoImage(Image.open("./images/Pinky_BottomRight.png").resize((WIDTH//4, HEIGHT//4)))
}

posture_corrections = [[flat_fingers, raise_wrists], [flying_pinkie, thumb_falling]]


# GRID.
for i in range(2):
    root.columnconfigure(i, weight=1, minsize=WIDTH//4)
    root.rowconfigure(i, weight=1, minsize=HEIGHT//4)

    for j in range(2):
        frame = tk.Frame(
            master=root,
            relief=tk.RAISED,
            borderwidth=1
        )

        frame.grid(row=i, column=j)
        error_label = tk.Label(image=posture_corrections[i][j]["image"], master=frame)
        image_label = tk.Label(image=posture_corrections[i][j]["image"], master=frame)
        error_label.pack()



root.mainloop()


## OTHER CODE.
# for i in range(len(posture_corrections)):
#     posture_corrections[i].pack()
#     frames[i].pack()