import tkinter as tk
from PIL import ImageTk, Image

# SETUP.
root = tk.Tk()
WIDTH = 1600
HEIGHT = 900
root.geometry("1600x900")

# Errors.
flat_fingers = {
    "text": "Lift Your Fingers!",
    "image": ImageTk.PhotoImage(Image.open("./images/Wrist_TopLeft.png").resize(((WIDTH//2, HEIGHT//2))))
}

raise_wrists = {
    "text": "Raise Your Wrists!",
    "image": ImageTk.PhotoImage(Image.open("./images/Fingers_TopRight.png").resize((WIDTH//2, HEIGHT//2)))
}

flying_pinkie = {
    "text": "Bend Your Pinky!",
    "image": ImageTk.PhotoImage(Image.open("./images/Thumb_BottomLeft.png").resize((WIDTH//2, HEIGHT//2)))
}

thumb_falling = {
    "text": "Put Your Thumb on the Keyboard!",
    "image": ImageTk.PhotoImage(Image.open("./images/Pinky_BottomRight.png").resize((WIDTH//2, HEIGHT//2)))
}

posture_corrections = [[flat_fingers, raise_wrists], [flying_pinkie, thumb_falling]]


# PLACE THE FRAMES.
frame = tk.Frame(root, width=WIDTH, height=HEIGHT)
frame.place(anchor="nw", x=0, y=0)

# IMAGES.
def display_error(errors:tuple):
    '''Display the errors of the user.'''

    if errors[0]:
        flat_fingers_label = tk.Label(image=posture_corrections[0][0]["image"], master=frame)
        flat_fingers_label.place(anchor="nw", x=0, y=0)

    if errors[1]:
        raise_wrists_label = tk.Label(image=posture_corrections[0][1]["image"], master=frame)
        raise_wrists_label.place(anchor="nw", x=WIDTH//2, y=0)

    if errors[2]:
        flying_pinkie_label = tk.Label(image=posture_corrections[1][0]["image"], master=frame)
        flying_pinkie_label.place(anchor="nw", x=0, y=HEIGHT//2)

    if errors[3]:
        thumb_falling_label = tk.Label(image=posture_corrections[1][1]["image"], master=frame)
        thumb_falling_label.place(anchor="nw", x=WIDTH//2, y=HEIGHT//2)

    # USE A PRINT-OUT. NOT SUPPORTED.
    # dinosaur_label = tk.Label(image=ImageTk.PhotoImage(Image.open("./images/Dino_Centre.png").resize((WIDTH//8, HEIGHT//8))), master=frame)
    # dinosaur_label.place(anchor="center", relx=0.5, rely=0.5)

display_error((True, True, True, True))


root.mainloop()
