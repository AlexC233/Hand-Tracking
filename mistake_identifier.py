import numpy as np
# Notes
'''
You can import any required modules required for the functions.
You can define additional functions as needed.
These functions are intended to be used in a larger program within a loop.
You should use the provided function signatures. (But if you really need to change them, let me know.)
You may use global variables if needed.
Your code should not execute anything if the file is imported as a module.
'''

# Input Specifications
# Joints Specifications
'''
Joint Indices   Joint
0               WRIST
1               THUMB CMC
2               THUMB MCP
3               THUMB IP
4               THUMB TIP
5               INDEX MCP
6               INDEX PIP
7               INDEX DIP
8               INDEX TIP
9               MIDDLE MCP
10              MIDDLE PIP
11              MIDDLE DIP
12              MIDDLE TIP
13              RING MCP
14              RING PIP
15              RING DIP
16              RING TIP
17              PINKY MCP
18              PINKY PIP
19              PINKY DIP
20              PINKY TIP
See: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
'''
# Angles Specifications
'''
Angle Indices   Angle
0               Thumb MCP (Joints 1-2-3)
1               Thumb IP (Joints 2-3-4)
2               Index MCP (Joints 5-6-7)
3               Index PIP (Joints 6-7-8)
4               Middle MCP (Joints 9-10-11)
5               Middle PIP (Joints 10-11-12)
6               Ring MCP (Joints 13-14-15)
7               Ring PIP (Joints 14-15-16)
8               Pinky MCP (Joints 17-18-19)
9               Pinky PIP (Joints 18-19-20)
10              Angle Between Palm and Keyboard
'''

# Function 1
# Input
'''
You will be provided with a (21 x 3) numpy array. Each "row" will contain the 3d coordinates of the 21 joints outlined above.
Each coordinate will be a numpy array of 3 floats.
You will also be provided with a numpy array of 10 floats representing the angles outlined above in degrees.
'''
# Output
'''
You will be required to return a tuple of 4 booleans, with each boolean representing whether one of four mistakes has been made.
The mistakes are as follows (the numbers correspond to the index of the boolean in the tuple):
0. Fingers are flat.
1. Wrists are too low.
2. Excessive tension in the pinky (pinky too high/stright).
3. Thumb is too low.
See: https://docs.google.com/document/d/158R2p73HYC6tiNJWN3kmQfs1sC8OL2WKY25BcG7B2iY/edit#heading=h.1iffdz15pj8z
'''
def mistake_identifier(coordinates:np.array, angles:np.array)->tuple:
    '''Return a tuple of 4 booleans, with each boolean representing whether one of four mistakes has been made.
    True means the mistake has been made, False means it has not been made.'''
    # Assume all mistakes are being made.
    result = [True, False, False, False]
    
    # Note: More negative z-values are higher up.
    
    # check for flat fingers
    # check if ALL of angles 2-9 are within 6.6 degrees of 180 degrees.
    for i in range(2, 10):
        if np.abs(angles[i] - 180) > 6.6:
            result[0] = False
            break

    # Wrists are too low 
    # check if the wrist is below the thumb tip or the angle of elevation is greater than 45 degrees.
    # since the wrist is the origin, we can just check the z-coordinate of the thumb tip.
    if angles[10] < -18:
        result[1] = True
        
    # Excessive tension in the pinky
    # check if the pinky tip is above the index, middle, or ring tip and the pinky joint angles are nearly 180 degrees.
    for i in [8, 12, 16]:
        if coordinates[20][2] < coordinates[i][2] - 0.03:
            result[2] = True
            break
    for i in range(2):
        if np.abs(angles[8 + i] - 180) > 10:
            result[2] = False
            break    

    # Thumb is too low.
    # if the thumb is below the index finger.
    if coordinates[4][2] > coordinates[8][2] + 0.03:
        result[3] = True

    return tuple(result)


# Bonus
'''
We likely want to only alert the user if a mistake is made for a certain amount of time.
This will prevent the user from being bombarded with alerts.
'''
# Input
'''
errors: The tuple of 4 booleans from the previous function.
threshold: The amount of time in seconds that the mistake must be made.
'''
# Output
'''
A tuple of 4 booleans, with each boolean representing whether one of four mistakes has been made for the specified amount of time.
'''


import time

# Each entry corresponds to the time in which the corresponding error (as specified with the indices above) occurred.
start_time = time.time()
time_errors_made = [start_time, start_time, start_time, start_time]


# Use additional functions as needed
# You may need to use a global variable to store the time the mistake was made
def alert_needed(errors:tuple, threshold:int)->tuple:
    global time_errors_made
    
    # Obtain the current time and whether an alert is required.
    current_time = time.time()
    need_alert = [False, False, False, False]

    # Assume the error was passed into the alert_needed function when the error occurred.
    for i in range(len(errors)):
        if errors[i]:
            time_errors_made[i] = time.time()
        
        if errors[i] and current_time - time_errors_made[i] > threshold:
            need_alert[i] = True
        else:
            need_alert[i] = False
    
    return tuple(need_alert)