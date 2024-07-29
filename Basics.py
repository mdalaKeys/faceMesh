import cv2
import time
from FaceMeshModule import FaceMeshDetector

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)
pTime = 0  # Initialize previous time
detector = FaceMeshDetector(maxFaces=2)  # Initialize face mesh detector

while True:
    success, img = cap.read()  # Read frame from the camera
    if not success:
        break  # Exit loop if frame not read successfully

    # Detect face meshes and get landmarks
    img, faces = detector.findFaceMesh(img)

    # Display face indices
    if faces:  # Ensure faces is not empty
        for idx, face in enumerate(faces):
            cv2.putText(
                img,
                f'Face {idx}',
                (20, 40 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )

    # Calculate FPS and display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the image with annotations
    cv2.imshow("Face Mesh Detection", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
