import tkinter as tk

# SETUP.
root = tk.Tk()


# Errors.
flat_fingers = "Lift Your Fingers!"
raise_wrists = "Raise Your Wrists!"
flying_pinkie = "Bend Your Pinky!"
thumb_falling = "Put Your Thumb on the Keyboard!"
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
        error_label = tk.Label(text=posture_corrections[i][j], master=frame)
        posture_corrections[i][j]
        error_label.pack(padx=5, pady=5)



root.mainloop()


## OTHER CODE.
# for i in range(len(posture_corrections)):
#     posture_corrections[i].pack()
#     frames[i].pack()