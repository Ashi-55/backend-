import cv2
from flask import Flask, Response, render_template_string, request, jsonify
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import numpy as np
import threading
import queue
import speech_recognition as sr
import pyttsx3
import os

app = Flask(__name__)

# YOLOS-Tiny setup
model_name = "hustvl/yolos-tiny"
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Voice setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Ensure espeak is used for AWS EC2
engine.setProperty('rate', 150)
engine.setProperty('voice', 'english')
engine.setProperty('volume', 1.0)

command_queue = queue.Queue()

# 🔹 Check for a real webcam (EC2 won't have one)
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No webcam detected")
except:
    print("❌ No webcam available. Using placeholder.")

# 🔹 Function to analyze image using YOLOS model
def analyze_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits.softmax(-1)[0]
    max_score_idx = scores.max(dim=0).indices[-1]
    confidence = scores.max(dim=0).values[-1].item()
    
    label = model.config.id2label[max_score_idx.item()] if confidence > 0.5 else "Nothing detected"
    response = f"I see: {label}"

    # 🔹 Text-to-Speech
    engine.say(response)
    engine.runAndWait()

    return label, response

# 🔹 Function to generate a camera feed (EC2 uses a placeholder)
def generate_frames():
    if cap is None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera on EC2", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 🔹 Function to process voice commands from a file (EC2 workaround)
def process_audio_file(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error: {e}"

# 🔹 Routes
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Camera AI Assistant</title>
    </head>
    <body>
        <h1>Camera AI Assistant</h1>
        <img src="{{ url_for('video_feed') }}" id="video-feed" alt="Live Feed">
        <button id="capture-btn">Capture & Analyze</button>
        <p>Upload an audio file for voice commands.</p>
        <input type="file" id="audio-input" accept="audio/wav">
        <button id="process-audio-btn">Process Audio</button>
        <div id="response">
            <h3>AI Description:</h3>
            <p id="description">Waiting for capture...</p>
            <h3>AI Response:</h3>
            <p id="response-text">Waiting for response...</p>
        </div>
        <script>
            document.getElementById('capture-btn').addEventListener('click', function() {
                fetch('/capture', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            alert(data.error);
                        } else {
                            document.getElementById('description').textContent = data.description;
                            document.getElementById('response-text').textContent = data.response;
                        }
                    });
            });

            document.getElementById('process-audio-btn').addEventListener('click', function() {
                var fileInput = document.getElementById('audio-input').files[0];
                if (!fileInput) {
                    alert("Please select an audio file.");
                    return;
                }
                var formData = new FormData();
                formData.append('file', fileInput);

                fetch('/process_audio', { method: 'POST', body: formData })
                    .then(response => response.json())
                    .then(data => {
                        alert("Command: " + data.command);
                    });
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    ret, frame = cap.read() if cap else (True, np.zeros((240, 320, 3), dtype=np.uint8))
    if not ret:
        return jsonify({'error': 'Failed to capture image'})
    
    description, response = analyze_image(frame)
    return jsonify({'description': description, 'response': response})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)

    command = process_audio_file(file_path)
    os.remove(file_path)  # Clean up the file after processing

    return jsonify({'command': command})

# 🔹 Run Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
    finally:
        if cap:
            cap.release()
