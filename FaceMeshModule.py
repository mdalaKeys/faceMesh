import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initializes the FaceMeshDetector.

        Args:
            staticMode (bool): Whether to detect static images only.
            maxFaces (int): Maximum number of faces to detect.
            minDetectionCon (float): Minimum confidence threshold for face detection.
            minTrackCon (float): Minimum confidence threshold for face tracking.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Initialize MediaPipe FaceMesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )

        # Drawing specification for landmarks
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Detects face landmarks in the input image.

        Args:
            img (numpy.ndarray): Input image in BGR format.
            draw (bool): Whether to draw the landmarks on the image.

        Returns:
            numpy.ndarray: Image with or without landmarks drawn.
            list: List of detected faces, each represented as a list of landmarks.
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec
                    )
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec
                    )
                face = []
                for lm in faceLms.landmark:
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)

        return img, faces
