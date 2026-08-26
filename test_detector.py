import cv2

from detector import BananaDetector


def main():

    detector = BananaDetector(
        model_path="yolo11n.pt",
        confidence=0.4
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Erro ao abrir webcam.")

        return


    while True:

        ret, frame = camera.read()

        if not ret:
            break


        detections = detector.detect(
            frame
        )


        for detection in detections:

            x1, y1, x2, y2 = \
                detection["bbox"]

            confidence = \
                detection["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Banana {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        cv2.imshow(
            "Detector de Bananas",
            frame
        )


        if cv2.waitKey(1) & 0xFF == 27:
            break


    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()