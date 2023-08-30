import cv2

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)

CLASS_FILE = "coco.names"
CONFIG_PATH = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WEIGHTS_PATH = 'frozen_inference_graph.pb'
TEST_OBJECT = 'person'

cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

def show_distance(img, box, screen_center_x, screen_center_y):
    obj_center_x = (box[0] + box[2]) // 2
    obj_center_y = (box[1] + box[3]) // 2
    distance_x = obj_center_x - screen_center_x
    distance_y = obj_center_y - screen_center_y
    
    text = f"Dist X: {distance_x}, Dist Y: {distance_y}"
    cv2.putText(img, text, (box[0] + 10, box[1] - 10),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.circle(img, (obj_center_x, obj_center_y), 10, (0, 255, 0), -1)
    cv2.circle(img, (screen_center_x, screen_center_y), 10, (0, 0, 255))

def add_one_object(img, box, current_obj, confidence):
    color = (0, 255, 0)
    cv2.rectangle(img, box, color, thickness=2)
    cv2.putText(
        img, current_obj.upper(), (box[0] + 10, box[1] + 30),
        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

def main():
    classNames = []

    with open(CLASS_FILE, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    #net.setInputSwapRB(True)

    entered_name=TEST_OBJECT

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            screen_center_x = img.shape[1] // 2
            screen_center_y = img.shape[0] // 2

            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classNames[classId - 1].lower() == entered_name:
                    add_one_object(img, box, classNames[classId - 1], confidence)
                    show_distance(img, box, screen_center_x, screen_center_y)
                    break

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
