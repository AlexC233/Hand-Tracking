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

if __name__ == "__main__":
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    
    (length, width) = (640, 480)
    
    # set the cameras to 640x480
    cam1.set(3, length)
    cam1.set(4, width)
    cam2.set(3, length)
    cam2.set(4, width)
    
    inverted = False

    raw = {"cam1":
        {"use": True, "cam": cam1, "image": None},
        "cam2":
        {"use": False, "cam": cam2, "image": None},
    }

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

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    
    while True:
        for key in raw:
            if raw[key]["use"]:
                success, img = raw[key]["cam"].read()
                raw[key]["image"] = img
            else:
                raw[key]["image"] = np.zeros((480, 640, 3), np.uint8)
            
        results["cam1"]["lms"] = hands.process(cv2.cvtColor(raw["cam1"]["image"], cv2.COLOR_BGR2RGB))
        results["cam2"]["lms"] = hands.process(cv2.cvtColor(raw["cam2"]["image"], cv2.COLOR_BGR2RGB))
        
        try:
            for key in results:
                if not raw[key]["use"]:
                    continue
                if results[key]["lms"].multi_hand_landmarks:
                    # check the handeness
                    handedness = results[key]["lms"].multi_handedness[0].classification[0].label
                    handedness = handedness.lower()
                    if inverted:
                        if handedness == "left":
                            handedness = "right"
                        else:
                            handedness = "left"
                    if handedness == "left":
                        results[key]["left"]["detected"] = True
                    else:
                        results[key]["right"]["detected"] = True
                        
                    # reset the coordinates and screen positions and angles
                    results[key][handedness]["coordinates"] = np.empty((0, 3))
                    results[key][handedness]["screen_pos"] = np.empty((0, 2))
                    results[key][handedness]["angles"] = np.empty(0)
        
                    for id, lm in enumerate(results[key]["lms"].multi_hand_landmarks[0].landmark):
                        h, w, c = raw[key]["image"].shape
                        cx, cy = int(lm.x *w), int(lm.y*h)
                        results[key][handedness]["screen_pos"] = np.vstack((results[key][handedness]["screen_pos"], (cx, cy)))
                        cv2.circle(raw[key]["image"], (cx,cy), 3, (255,0,255), cv2.FILLED)
                        # get the joint coordinates
                        coord = np.array([lm.x, lm.y, lm.z])
                        # display rounded coordinates
                        # rcoord = tuple(map(lambda x: round(x, 2), coord))
                        # cv2.putText(img, str(id) + " " + str(rcoord), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        results[key][handedness]["coordinates"] = np.vstack((results[key][handedness]["coordinates"], coord))
                    
                    results[key][handedness]["angles"] = get_all_angles(results[key][handedness]["coordinates"])
                    
                    # put the angles on the image at their respective joints
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][0], 1)), (int(results[key][handedness]["screen_pos"][2] [0]), int(results[key][handedness]["screen_pos"][2] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb MCP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][1], 1)), (int(results[key][handedness]["screen_pos"][3] [0]), int(results[key][handedness]["screen_pos"][3] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb IP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][2], 1)), (int(results[key][handedness]["screen_pos"][6] [0]), int(results[key][handedness]["screen_pos"][6] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index MCP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][3], 1)), (int(results[key][handedness]["screen_pos"][7] [0]), int(results[key][handedness]["screen_pos"][7] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index PIP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][4], 1)), (int(results[key][handedness]["screen_pos"][10][0]), int(results[key][handedness]["screen_pos"][10][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle MCP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][5], 1)), (int(results[key][handedness]["screen_pos"][11][0]), int(results[key][handedness]["screen_pos"][11][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle PIP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][6], 1)), (int(results[key][handedness]["screen_pos"][14][0]), int(results[key][handedness]["screen_pos"][14][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring MCP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][7], 1)), (int(results[key][handedness]["screen_pos"][15][0]), int(results[key][handedness]["screen_pos"][15][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring PIP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][8], 1)), (int(results[key][handedness]["screen_pos"][18][0]), int(results[key][handedness]["screen_pos"][18][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky MCP 
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][9], 1)), (int(results[key][handedness]["screen_pos"][19][0]), int(results[key][handedness]["screen_pos"][19][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky PIP
                    cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][10], 1)), (int(results[key][handedness]["screen_pos"][0][0]), int(results[key][handedness]["screen_pos"][0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Wrist Elevation

                        
                    if len(results[key]["lms"].multi_hand_landmarks) == 2:
                        
                        # check the handeness
                        handedness = results[key]["lms"].multi_handedness[1].classification[0].label
                        handedness = handedness.lower()
                        if inverted:
                            if handedness == "left":
                                handedness = "right"
                            else:
                                handedness = "left"
                        if handedness == "left":
                            results[key]["left"]["detected"] = True
                        else:
                            results[key]["right"]["detected"] = True
                            
                        # reset the coordinates and screen positions and angles
                        results[key][handedness]["coordinates"] = np.empty((0, 3))
                        results[key][handedness]["screen_pos"] = np.empty((0, 2))
                        results[key][handedness]["angles"] = np.empty(0)
                        
                        for id, lm in enumerate(results[key]["lms"].multi_hand_landmarks[1].landmark):
                            h, w, c = raw[key]["image"].shape
                            cx, cy = int(lm.x *w), int(lm.y*h)
                            results[key][handedness]["screen_pos"] = np.vstack((results[key][handedness]["screen_pos"], (cx, cy)))
                            cv2.circle(raw[key]["image"], (cx,cy), 3, (255,0,255), cv2.FILLED)
                            # get the joint coordinates
                            coord = np.array([lm.x, lm.y, lm.z])
                            # display rounded coordinates
                            # rcoord = tuple(map(lambda x: round(x, 2), coord))
                            # cv2.putText(img, str(id) + " " + str(rcoord), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            results[key][handedness]["coordinates"] = np.vstack((results[key][handedness]["coordinates"], coord))
                                    
                        results[key][handedness]["angles"] = get_all_angles(results[key][handedness]["coordinates"])
                        
                        # put the angles on the image at their respective joints
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][0], 1)), (int(results[key][handedness]["screen_pos"][2] [0]), int(results[key][handedness]["screen_pos"][2] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb MCP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][1], 1)), (int(results[key][handedness]["screen_pos"][3] [0]), int(results[key][handedness]["screen_pos"][3] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Thumb IP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][2], 1)), (int(results[key][handedness]["screen_pos"][6] [0]), int(results[key][handedness]["screen_pos"][6] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index MCP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][3], 1)), (int(results[key][handedness]["screen_pos"][7] [0]), int(results[key][handedness]["screen_pos"][7] [1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Index PIP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][4], 1)), (int(results[key][handedness]["screen_pos"][10][0]), int(results[key][handedness]["screen_pos"][10][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle MCP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][5], 1)), (int(results[key][handedness]["screen_pos"][11][0]), int(results[key][handedness]["screen_pos"][11][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Middle PIP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][6], 1)), (int(results[key][handedness]["screen_pos"][14][0]), int(results[key][handedness]["screen_pos"][14][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring MCP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][7], 1)), (int(results[key][handedness]["screen_pos"][15][0]), int(results[key][handedness]["screen_pos"][15][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Ring PIP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][8], 1)), (int(results[key][handedness]["screen_pos"][18][0]), int(results[key][handedness]["screen_pos"][18][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky MCP 
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][9], 1)), (int(results[key][handedness]["screen_pos"][19][0]), int(results[key][handedness]["screen_pos"][19][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Pinky PIP
                        cv2.putText(raw[key]["image"], str(np.round(results[key][handedness]["angles"][10], 1)), (int(results[key][handedness]["screen_pos"][0][0]), int(results[key][handedness]["screen_pos"][0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1) # Wrist Elevation
                else:
                    results[key]["left"]["detected"] = False
                    results[key]["right"]["detected"] = False
        except KeyboardInterrupt:
            for key in raw:
                raw[key]["cam"].release()
            cv2.destroyAllWindows()
            break                    
        except Exception as e:
            print(e)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        for key in raw:
            cv2.putText(raw[key]["image"], str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            cv2.imshow(key, raw[key]["image"])
            cv2.waitKey(1)