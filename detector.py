from ultralytics import YOLO
import cv2


class BananaDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.5
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence


    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0].cpu().numpy()
                )

                class_name = \
                    self.model.names[class_id]

                if class_name.lower() != "banana":
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = float(
                    box.conf[0].cpu().numpy()
                )

                detections.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "confidence": confidence
                })

        return detections