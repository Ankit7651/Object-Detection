# Detecting Objects on Video with OpenCV deep learnng library

# Importing needed libraries
import numpy as np
import cv2
import time


"""
Start of:
Reading input video
"""


video = cv2.VideoCapture('videos/Pexels Videos 2034115.mp4')

"""
End of:
Reading input video
"""


"""
Start of:
Loading YOLO v3 network
"""


with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# Check point
print('List with labels names:')
print(labels)


network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')


layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""

# Defining variable for counting frames

f = 0
t = 0

# Defining loop for catching frames
while True:

    if not ret:
        break

    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    """
    Start of:
    Getting blob from current frame
    """

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

   
    f += 1
    t += end - start

    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
     
        for detected_objects in result:
           
            scores = detected_objects[5:]
       
            class_current = np.argmax(scores)
      
            confidence_current = scores[class_current]



           
            if confidence_current > probability_minimum:
               
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """


    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """


    if len(results) > 0:

        for i in results.flatten():

            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()

            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 5)


            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Writing processed frame into the file
    """

    if writer is None:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')


        writer = cv2.VideoWriter('videos/result-traffic-cars8.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

    """
    End of:
    Writing processed frame into the file
    """

"""
End of:
Reading frames in the loop
"""


# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()


