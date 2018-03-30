import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_boxes(boxes, frame):

    frame2 = frame.copy()

    for box in boxes:
        frame2 = cv2.rectangle(frame2,
                               (int(box[0]), int(box[1])),
                               (int(box[0]+box[2]), int(box[1]+box[3])),
                               (255,255,0),
                               2)

    return frame2
