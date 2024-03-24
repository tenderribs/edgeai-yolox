import cv2
import os
import json

# supposed names of the keypoints
keypoint_names = ["kpC", "kpL", "kpR", "AX1", "AX2"]

with open('./datasets/SynthTree43k/annotations/trees_val.json') as json_file:
    data = json.load(json_file)
    image = data["images"][3]

    id = image['id']

    # get a list of annotations for the image
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == id]

    image_filename = os.path.join("./datasets/SynthTree43k", image['file_name'])
    cvimage = cv2.imread(image_filename)
    for tree in annotations:
        bbox = tree['bbox']
        x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        p1 = (x,y)
        p2 = (x + w, y + h)

        cv2.rectangle(cvimage, p1, p2, (0,0,255), 2)

        kp = tree['keypoints']
        for idx in range(5):
            x = int(kp[0 + idx*3])
            y = int(kp[1 + idx*3])

            point = (x, y)
            label_pos = (x - 40, y + 5)

            label = keypoint_names[idx]

            cv2.circle(cvimage, point, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.putText(cvimage, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fname = f"./YOLOX_outputs/{image['file_name']}"
    cv2.imwrite(fname, cvimage)
    print(f"Saved overlaid image to {fname}")