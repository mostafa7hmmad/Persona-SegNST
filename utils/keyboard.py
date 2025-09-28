import cv2
from config import GROUPS, KEY_W, KEY_H, START_X, START_Y

current_mode = "Style"
current_button = "Candy"

def draw_keyboard(frame):
    global current_button
    x_offset = START_X
    for group, keys in GROUPS.items():
        cv2.putText(frame, group, (x_offset, START_Y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for key in keys:
            x, y = x_offset, START_Y
            if key == current_button:
                cv2.rectangle(frame, (x, y), (x + KEY_W, y + KEY_H), (0, 200, 0), -1)
                cv2.putText(frame, key, (x + 3, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                cv2.rectangle(frame, (x, y), (x + KEY_W, y + KEY_H), (0, 255, 0), 1)
                cv2.putText(frame, key, (x + 3, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x_offset += KEY_W + 5
        x_offset += 15
    return frame


def mouse_click(event, x, y, flags, param):
    global current_style, current_mode, current_button
    loaded_styles = param["loaded_styles"]
    style_models = param["style_models"]

    if event == cv2.EVENT_LBUTTONDOWN:
        x_offset = START_X
        for group, keys in GROUPS.items():
            for key in keys:
                kx, ky = x_offset, START_Y
                if kx <= x <= kx + KEY_W and ky <= y <= ky + KEY_H:
                    current_button = key
                    if key in style_models:
                        current_style = loaded_styles[key]
                        current_mode = "Style"
                    else:
                        current_mode = key
                x_offset += KEY_W + 5
            x_offset += 15
