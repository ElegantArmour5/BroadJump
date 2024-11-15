from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_video(video_path):
    # Save output in the uploads folder
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_path).split('.')[0] + "_output.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_point, end_point = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_heel = landmarks[29]
            right_heel = landmarks[31]

            heel_x = int((left_heel.x + right_heel.x) / 2 * width)
            heel_y = int((left_heel.y + right_heel.y) / 2 * height)

            cv2.circle(frame, (heel_x, heel_y), 5, (0, 255, 0), -1)

            if start_point is None:
                start_point = (heel_x, heel_y)
            else:
                end_point = (heel_x, heel_y)

        if start_point:
            cv2.circle(frame, start_point, 10, (255, 0, 0), -1)
            cv2.putText(frame, "Start", (start_point[0] + 10, start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if end_point:
            cv2.circle(frame, end_point, 10, (0, 0, 255), -1)
            cv2.putText(frame, "End", (end_point[0] + 10, end_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    if start_point and end_point:
        pixel_distance = int(((end_point[0] - start_point[0])**2 + 
                              (end_point[1] - start_point[1])**2) ** 0.5)
        return pixel_distance

    return None


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        video = request.files['video']
        if video.filename == '':
            return redirect(request.url)

        if video:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)

            distance = process_video(filepath)
            if distance is not None:
                return render_template('index.html', distance=distance)

    return render_template('index.html', distance=None)



if __name__ == '__main__':
    app.run(debug=True)
