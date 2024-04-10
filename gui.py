import tkinter as tk
from PIL import ImageTk, Image

# SETUP.
root = tk.Tk()
#canvas = tk.Canvas(root, width=root.winfo_width(), height=root.winfo_height())


# Errors.
flat_fingers = {
    "text": "Lift Your Fingers!",
    "image": ImageTk.PhotoImage(Image.open("./images/raise_wrists.png"))
}

raise_wrists = {
    "text": "Raise Your Wrists!",
    "image": ImageTk.PhotoImage(Image.open("./images/raise_wrists.png"))
}

flying_pinkie = {
    "text": "Bend Your Pinky!",
    "image": ImageTk.PhotoImage(Image.open("./images/raise_wrists.png"))
}

thumb_falling = {
    "text": "Put Your Thumb on the Keyboard!",
    "image": ImageTk.PhotoImage(Image.open("./images/raise_wrists.png"))
}

posture_corrections = [[flat_fingers, raise_wrists], [flying_pinkie, thumb_falling]]


# GRID.
for i in range(2):
    root.columnconfigure(i, weight=1, minsize=root.winfo_screenwidth()/2)
    root.rowconfigure(i, weight=1, minsize=root.winfo_screenheight()/2)
    
    for j in range(2):
        frame = tk.Frame(
            master=root,
            relief=tk.RAISED,
            borderwidth=1
        )

        frame.grid(row=i, column=j, padx=5, pady=5)
        error_label = tk.Label(image=posture_corrections[i][j]["image"], master=frame)
        image_label = tk.Label(image=posture_corrections[i][j]["image"], master=frame)
        error_label.pack(padx=5, pady=5)



root.mainloop()


## OTHER CODE.
# for i in range(len(posture_corrections)):
#     posture_corrections[i].pack()
#     frames[i].pack()