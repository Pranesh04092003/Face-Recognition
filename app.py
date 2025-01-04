from flask import Flask, jsonify, Response, render_template
import face_recognition
import cv2
import numpy as np
import pickle
import threading

app = Flask(__name__)

# Load encoded dataset
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]

output_frame = None
lock = threading.Lock()
recognized_names = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognized_names')
def get_recognized_names():
    global recognized_names
    return jsonify({"names": recognized_names})

def generate_frames():
    global output_frame, lock, recognized_names
    video_capture = cv2.VideoCapture(0)

    # Reduce frame size for faster processing
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Skip frames to improve speed
        if frame_count % 2 == 0:
            # Convert to grayscale for detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization
            equalized_frame = cv2.equalizeHist(gray_frame)

            # Detect faces on the equalized frame
            face_locations = face_recognition.face_locations(equalized_frame)

            # Use the original frame for encoding
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            recognized_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                recognized_names.append(name)

            # Draw rectangles and names
            for (top, right, bottom, left), name in zip(face_locations, recognized_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with lock:
                output_frame = frame.copy()

        frame_count += 1

        (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

    video_capture.release()

if __name__ == '__main__':
    app.run(debug=True)