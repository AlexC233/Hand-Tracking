import tkinter as tk
from PIL import ImageTk, Image
import track
import playsound # you might encounter an error here, if you do, run `pip install playsound` in your terminal.
# if `pip install playsound` shows an error about some "wheel" thing, run `pip install --upgrade wheel` first and then try again
import time

# SETUP.
root = tk.Tk()
WIDTH = 1600
HEIGHT = 900
root.geometry("1600x900")

last_time = time.time()
sound = "./audio/error.wav"

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

dinosaur = ImageTk.PhotoImage(Image.open("./images/Dino_Centre.png").resize((WIDTH//8, HEIGHT//8)))

posture_corrections = [[flat_fingers, raise_wrists], [flying_pinkie, thumb_falling]]


# PLACE THE FRAMES.
frame = tk.Frame(root, width=WIDTH, height=HEIGHT)
frame.place(anchor="nw", x=0, y=0)

# IMAGES.
def display_error(errors:tuple):
    '''Display the errors of the user.'''
    
    # First clear the frame.
    for widget in frame.winfo_children():
        widget.destroy()
        
    if errors != (False, False, False, False):
        play_sound()
        
    display = tk.Canvas(frame, width=WIDTH, height=HEIGHT)
    display.place(anchor="nw", x=0, y=0)
        
    # for some reason these are reversed.
    if errors[1]: # low wrist
        # flat_fingers_label = tk.Label(image=posture_corrections[0][0]["image"], master=frame)
        # flat_fingers_label.place(anchor="nw", x=0, y=0)
        display.create_image(0, 0, image=posture_corrections[0][0]["image"], anchor="nw")

    if errors[0]: # flat fingers
        # raise_wrists_label = tk.Label(image=posture_corrections[0][1]["image"], master=frame)
        # raise_wrists_label.place(anchor="nw", x=WIDTH//2, y=0)
        display.create_image(WIDTH//2, 0, image=posture_corrections[0][1]["image"], anchor="nw")

    if errors[3]: # low thumb
        # flying_pinkie_label = tk.Label(image=posture_corrections[1][0]["image"], master=frame)
        # flying_pinkie_label.place(anchor="nw", x=0, y=HEIGHT//2)
        display.create_image(0, HEIGHT//2, image=posture_corrections[1][0]["image"], anchor="nw")

    if errors[2]: # flying pinkie
        # thumb_falling_label = tk.Label(image=posture_corrections[1][1]["image"], master=frame)
        # thumb_falling_label.place(anchor="nw", x=WIDTH//2, y=HEIGHT//2)
        display.create_image(WIDTH//2, HEIGHT//2, image=posture_corrections[1][1]["image"], anchor="nw")
        
    # put the dinosaur in the center
    display.create_image(WIDTH//2, HEIGHT//2, image=dinosaur, anchor="center")

def play_sound():
    '''Play a sound if it is not already playing.'''
    global last_time
    if time.time() - last_time >= 0.9:
        try:
            playsound.playsound(sound, block=False)
            
        except Exception as e:
            print(e)
        last_time = time.time()


display_error((False, False, False, False))

# every 100ms, check for errors.
def check_errors():
    '''Check for errors in the user's posture.'''
    try:
        errors = track.main()
    except Exception as e:
        print(e)
    display_error(errors)
    root.after(100, check_errors)
    
root.after(100, check_errors)

root.mainloop()
