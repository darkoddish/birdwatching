import cv2

def test_camera():
    camera = cv2.VideoCapture('/dev/video19')
    if not camera.isOpened():
        print("Camera not accessible")
        return

    print("Camera accessed successfully")
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
