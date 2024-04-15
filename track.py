# For Reference
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
'''
Input:
Two usb cameras will be fed into the program.
'''
'''
Output:
This program will output the 3d coordinates of the 21 joints outlined above as a tuple.
It will also output the following angles as a tuple:
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
10              Wrist Elevation
'''

import cv2
import mediapipe as mp
import time
import numpy as np
import mistake_identifier as mi

def get_angle(joint1:np.array, joint2:np.array, joint3:np.array) -> float:
    '''
    Calculate the angle between 3 joints
    '''
    # Calculate the vectors
    vector1 = joint1 - joint2
    vector2 = joint3 - joint2
    # Calculate the angle
    angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    return np.degrees(angle)

def get_all_angles(joints:np.array) -> np.array:
    output = np.array([])
    # Thumb
    output = np.append(output, get_angle(joints[1], joints[2], joints[3])) # Thumb MCP
    output = np.append(output, get_angle(joints[2], joints[3], joints[4])) # Thumb IP
    # Index
    output = np.append(output, get_angle(joints[5], joints[6], joints[7])) # Index MCP
    output = np.append(output, get_angle(joints[6], joints[7], joints[8])) # Index PIP
    # Middle
    output = np.append(output, get_angle(joints[9], joints[10], joints[11])) # Middle MCP
    output = np.append(output, get_angle(joints[10], joints[11], joints[12])) # Middle PIP
    # Ring
    output = np.append(output, get_angle(joints[13], joints[14], joints[15])) # Ring MCP
    output = np.append(output, get_angle(joints[14], joints[15], joints[16])) # Ring PIP
    # Pinky
    output = np.append(output, get_angle(joints[17], joints[18], joints[19])) # Pinky MCP
    output = np.append(output, get_angle(joints[18], joints[19], joints[20])) # Pinky PIP
    
    # To find the wrist elevation, we will first assume that the camera is facing the horizontal plane.
    # We will then find the angle between the wrist and the horizontal plane.
    # We will use the wrist as the origin and the index MCP as the point to find the angle.
    # find the vector from the wrist to the index MCP
    vector = joints[5] - joints[0]
    # find the angle between the vector and the horizontal plane
    output = np.append(output, 90 - np.degrees(np.arccos(vector[2] / np.linalg.norm(vector))))
    
    return output

# Setup code
cam1 = cv2.VideoCapture(0)
# cam2 = cv2.VideoCapture(2)

(length, width) = (640, 480)

# set the cameras to 640x480
# cam1.set(3, length)
# cam1.set(4, width)
# cam2.set(3, length)
# cam2.set(4, width)

# cam1 = None
cam2 = None

inverted = False

raw = {"cam1":
    {"use": True, "cam": cam1, "image": None},
    "cam2":
    {"use": False, "cam": cam2, "image": None},
}

cameras = [key for key in raw if raw[key]["use"]]

results = {"cam1": 
        {"lms": None,
        "left": 
            {"detected": False, "coordinates": np.empty((0, 3)), "screen_pos": np.empty((0, 2)), "angles": np.empty(0)},
        "right":
            {"detected": False, "coordinates": np.empty((0, 3)), "screen_pos": np.empty((0, 2)), "angles": np.empty(0)}
        },
    "cam2":
        {"lms": None,
        "left":
            {"detected": False, "coordinates": np.empty((0, 3)), "screen_pos": np.empty((0, 2)), "angles": np.empty(0)},
        "right":
            {"detected": False, "coordinates": np.empty((0, 3)), "screen_pos": np.empty((0, 2)), "angles": np.empty(0)},
        }
    }

merged_results = {"left":
    {"detected": False, "coordinates": np.empty((0, 3)), "angles": np.empty(0)},
    "right":
    {"detected": False, "coordinates": np.empty((0, 3)), "angles": np.empty(0)}
}

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

def get_image(cam:str)->None:
    if raw[cam]["use"]:
        success, img = raw[cam]["cam"].read()
        raw[cam]["image"] = img

def process_image(cam:str)->None:
    results[cam]["lms"] = hands.process(cv2.cvtColor(raw[cam]["image"], cv2.COLOR_BGR2RGB))
    
def find_handedness(cam:str, hand, handnum:int)->str:
    result = hand.multi_handedness[handnum].classification[0].label.lower()
    if inverted:
        if result == "left":
            return "right"
        else:
            return "left"
    return result

def get_coordinates(cam:str, hand, handedness, handnum:int)->None:
    for id, lm in enumerate(hand.multi_hand_landmarks[handnum].landmark):
        h, w, c = raw[cam]["image"].shape
        cx, cy = int(lm.x *w), int(lm.y*h)
        results[cam][handedness]["screen_pos"] = np.vstack((results[cam][handedness]["screen_pos"], np.array([cx, cy])))
        cv2.circle(raw[cam]["image"], (cx,cy), 3, (212,176,55), cv2.FILLED)
        # get the joint coordinates
        coord = np.array([lm.x, lm.y, lm.z])
        # display rounded coordinates
        # rcoord = tuple(map(lambda x: round(x, 2), coord))
        # cv2.putText(img, str(id) + " " + str(rcoord), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        results[cam][handedness]["coordinates"] = np.vstack((results[cam][handedness]["coordinates"], coord))
        
    # using world coordinates instead of screen coordinates
    # for id, lm in enumerate(hand.multi_hand_landmarks[handnum].landmark):
    #     h, w, c = raw[cam]["image"].shape
    #     cx, cy = int(lm.x *w), int(lm.y*h)
    #     results[cam][handedness]["screen_pos"] = np.vstack((results[cam][handedness]["screen_pos"], np.array([cx, cy])))
    #     cv2.circle(raw[cam]["image"], (cx,cy), 3, (255,0,255), cv2.FILLED)
    
    # for id, lm in enumerate(hand.multi_hand_world_landmarks[handnum].landmark):
    #     coord = np.array([lm.x, lm.y, lm.z])
    #     results[cam][handedness]["coordinates"] = np.vstack((results[cam][handedness]["coordinates"], coord))
        
def display_angles(cam:str, handedness)->None:
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][0] , 1)), (int(results[cam][handedness]["screen_pos"][2] [0]), int(results[cam][handedness]["screen_pos"][2] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb MCP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][1] , 1)), (int(results[cam][handedness]["screen_pos"][3] [0]), int(results[cam][handedness]["screen_pos"][3] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb IP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][2] , 1)), (int(results[cam][handedness]["screen_pos"][6] [0]), int(results[cam][handedness]["screen_pos"][6] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index MCP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][3] , 1)), (int(results[cam][handedness]["screen_pos"][7] [0]), int(results[cam][handedness]["screen_pos"][7] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index PIP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][4] , 1)), (int(results[cam][handedness]["screen_pos"][10][0]), int(results[cam][handedness]["screen_pos"][10][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle MCP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][5] , 1)), (int(results[cam][handedness]["screen_pos"][11][0]), int(results[cam][handedness]["screen_pos"][11][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle PIP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][6] , 1)), (int(results[cam][handedness]["screen_pos"][14][0]), int(results[cam][handedness]["screen_pos"][14][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring MCP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][7] , 1)), (int(results[cam][handedness]["screen_pos"][15][0]), int(results[cam][handedness]["screen_pos"][15][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring PIP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][8] , 1)), (int(results[cam][handedness]["screen_pos"][18][0]), int(results[cam][handedness]["screen_pos"][18][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky MCP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][9] , 1)), (int(results[cam][handedness]["screen_pos"][19][0]), int(results[cam][handedness]["screen_pos"][19][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky PIP
    cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["angles"][10], 1)), (int(results[cam][handedness]["screen_pos"][0] [0]), int(results[cam][handedness]["screen_pos"][0] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Wrist Elevation

def display_coordinates(cam:str, handedness:str, joints:list)->None:
    for i in joints:
        cv2.putText(raw[cam]["image"], str(np.round(results[cam][handedness]["coordinates"][i], 2)), (int(results[cam][handedness]["screen_pos"][i][0]), int(results[cam][handedness]["screen_pos"][i][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
 
def display_image(cam:str)->None:
    cv2.imshow(cam, raw[cam]["image"])
    cv2.waitKey(1)
    
def display_fps(cam:str)->None:
    global pTime
    global cTime
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(raw[cam]["image"], str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
def get_hand(cam:str, hand, handnum)->None:
    handedness = find_handedness(cam, hand, handnum)
    results[cam][handedness]["detected"] = True
    
    # reset the coordinates and screen positions and angles
    results[cam][handedness]["coordinates"] = np.empty((0, 3))
    results[cam][handedness]["screen_pos"] = np.empty((0, 2))
    results[cam][handedness]["angles"] = np.empty(0)
    
    get_coordinates(cam, hand, handedness, handnum)
    results[cam][handedness]["angles"] = get_all_angles(results[cam][handedness]["coordinates"])
    
    display_angles(cam, handedness)
    # display_coordinates(cam, handedness, [i for i in range(21)])
    
def process_hands(cam:str)->None:
    camlandmarks = results[cam]["lms"]
    # reset the detected hands
    results[cam]["left"]["detected"] = False
    results[cam]["right"]["detected"] = False
    if camlandmarks.multi_hand_landmarks:
        if len(camlandmarks.multi_hand_landmarks) == 1:
            get_hand(cam, camlandmarks, 0)
        elif len(camlandmarks.multi_hand_landmarks) == 2:
            get_hand(cam, camlandmarks, 0)
            get_hand(cam, camlandmarks, 1)
            
def run_track(cam:str)->None:
    get_image(cam)
    process_image(cam)
    process_hands(cam)
    # display_fps(cam)
    display_image(cam)
            
def track()->None:
    for cam in cameras:
        run_track(cam)

def distance_from_center(coord:np.array)->float:
    # return the distance from the center of the screen, which is (0.5, 0.5)
    return np.linalg.norm(coord[:2] - np.array([0.5, 0.5]))
        
def get_results()->None:
    ''' Process the results. Merge the results from all cameras.
        Take the average of the angles.
        Keep the coordinates from the camera with the readings closest to the center of the screen.
    '''
    # reset the merged results
    for hand in merged_results:
        merged_results[hand]["detected"] = False
        merged_results[hand]["coordinates"] = np.empty((0, 3))
        merged_results[hand]["angles"] = np.empty(0)
        
    # merge
    left_angles = np.empty((0, 11))
    right_angles = np.empty((0, 11))

    left_coords = np.empty((21, 3, 0))
    right_coords = np.empty((21, 3, 0))
    for cam in cameras:
        for hand in ("left", "right"):
            if results[cam][hand]["detected"]:
                if hand == "left":
                    left_angles = np.vstack((left_angles, results[cam][hand]["angles"]))
                    left_coords = np.dstack((left_coords, results[cam][hand]["coordinates"]))
                    
                else:
                    right_angles = np.vstack((right_angles, results[cam][hand]["angles"]))
                    right_coords = np.dstack((right_coords, results[cam][hand]["coordinates"]))
                    
    # average the angles
    if left_angles.size:
        merged_results["left"]["angles"] = np.mean(left_angles, axis=0)
        merged_results["left"]["detected"] = True
    if right_angles.size:
        merged_results["right"]["angles"] = np.mean(right_angles, axis=0)
        merged_results["right"]["detected"] = True
        
    # get the coordinates closest to the center of the screen
    if left_coords.size:
        left_distances = np.empty(left_coords.shape[2])
        for i in range(left_coords.shape[2]):
            sum = 0
            for j in range(21):
                sum += distance_from_center(left_coords[j, :, i])
            left_distances[i] = sum
            
        closest_left = np.argmin(left_distances)
        merged_results["left"]["coordinates"] = left_coords[:, :, closest_left]
    if right_coords.size:
        right_distances = np.empty(right_coords.shape[2])
        for i in range(right_coords.shape[2]):
            sum = 0
            for j in range(21):
                sum += distance_from_center(right_coords[j, :, i])
            right_distances[i] = sum
        
        closest_right = np.argmin(right_distances)
        merged_results["right"]["coordinates"] = right_coords[:, :, closest_right]          
        
    # test prints
    # check if a hand is detected
    # if merged_results["left"]["detected"]:
    #     print("Left Hand Detected")
    # if merged_results["right"]["detected"]:
    #     print("Right Hand Detected")
    
def posture_check()->tuple:
    ''' Check the posture of the hands. '''
    # if no hands are detected, return
    if not merged_results["left"]["detected"] and not merged_results["right"]["detected"]:
        return (False, False, False, False)
    left_result, right_result = (False, False, False, False), (False, False, False, False)
    # if the left hand is detected
    if merged_results["left"]["detected"]:
        left_result = mi.mistake_identifier(merged_results["left"]["coordinates"], merged_results["left"]["angles"])
    # if the right hand is detected
    if merged_results["right"]["detected"]:
        right_result = mi.mistake_identifier(merged_results["right"]["coordinates"], merged_results["right"]["angles"])

    # test print
    # print something if a mistake is detected
    # mistakes = ["Flat Fingers", "Raise Wrists", "Flying Pinkie", "Thumb Falling"]
    # for i in range(4):
    #     if left_result[i] or right_result[i]:
    #         print(mistakes[i])
    
    # if either hand has a mistake, then return the mistake
    return tuple([left_result[i] or right_result[i] for i in range(4)])
    
def main()->tuple:
    '''Main function. Returns the detected mistakes.'''
    track()
    get_results()
    return posture_check()
    

if __name__ == "__main__":    
    while True:
        try:
            main()
        except KeyboardInterrupt:
            for key in raw:
                raw[key]["cam"].release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)