from ultralytics import YOLO
import cv2

print("Loading YOLO model...")

model = YOLO("yolo26n.pt")

print("YOLO model loaded!")
print("Starting vehicle detection...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    results = model(frame, verbose=False)

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        if class_id == 2:
            cars += 1

        elif class_id == 3:
            motorcycles += 1

        elif class_id == 5:
            buses += 1

        elif class_id == 7:
            trucks += 1

    total = cars + motorcycles + buses + trucks

    detected_frame = results[0].plot()

    cv2.putText(
        detected_frame,
        f"Cars: {cars}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected_frame,
        f"Motorcycles: {motorcycles}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected_frame,
        f"Buses: {buses}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected_frame,
        f"Trucks: {trucks}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected_frame,
        f"Total Vehicles: {total}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Smart Traffic - Vehicle Counter",
        detected_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Vehicle detection stopped.")
