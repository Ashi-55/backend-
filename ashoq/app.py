import cv2
from flask import Flask, Response, render_template_string, request, jsonify
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import numpy as np
import threading
import queue
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

# YOLOS-Tiny setup
model_name = "hustvl/yolos-tiny"
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Voice setup
recognizer = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
command_queue = queue.Queue()

def listen_for_commands(command_queue):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Voice recognition started. Say 'capture' or 'quit'.")
        while True:
            try:
                audio = recognizer.listen(source, timeout=None)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                command_queue.put(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Voice recognition error: {e}")

voice_thread = threading.Thread(target=listen_for_commands, args=(command_queue,), daemon=True)
voice_thread.start()

# EC2 webcam workaround
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
except:
    print("No webcam available, using placeholder.")

# Image analysis
def analyze_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = rgb_frame
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(-1)[0]
    max_score_idx = scores.max(dim=0).indices[-1]
    confidence = scores.max(dim=0).values[-1].item()
    label = model.config.id2label[max_score_idx.item()] if confidence > 0.5 else "Nothing detected"
    response = f"I see: {label}"
    engine.say(response)
    engine.runAndWait()
    return label, response

# Camera feed
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

# Routes
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Fast Camera AI</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #74ebd5, #acb6e5);
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            h1 {
                color: #fff;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                font-size: 2.5em;
                margin-bottom: 20px;
            }
            #video-feed {
                width: 320px;
                height: 240px;
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            button {
                background-color: #ff6f61;
                color: white;
                border: none;
                padding: 12px 25px;
                font-size: 1.2em;
                border-radius: 25px;
                cursor: pointer;
                transition: transform 0.2s, background-color 0.2s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            button:hover {
                background-color: #ff4d3d;
                transform: scale(1.05);
            }
            #response {
                background: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                text-align: left;
            }
            h3 {
                color: #333;
                margin: 10px 0 5px;
            }
            p {
                color: #555;
                margin: 5px 0;
                font-size: 1.1em;
            }
        </style>
    </head>
    <body>
        <h1>Camera AI Assistant</h1>
        <img src="{{ url_for('video_feed') }}" id="video-feed" alt="Live Feed">
        <button id="capture-btn">Capture & Analyze</button>
        <p>Say "capture" to analyze or "quit" to stop.</p>
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
            function checkVoiceCommand() {
                fetch('/voice_command')
                    .then(response => response.json())
                    .then(data => {
                        if (data.command === 'capture') {
                            document.getElementById('description').textContent = data.description;
                            document.getElementById('response-text').textContent = data.response;
                        } else if (data.command === 'quit') {
                            alert('Quitting... Please close the tab.');
                        }
                    });
                setTimeout(checkVoiceCommand, 1000);
            }
            checkVoiceCommand();
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

@app.route('/voice_command', methods=['GET'])
def voice_command():
    try:
        command = command_queue.get_nowait()
        if "quit" in command:
            return jsonify({'command': 'quit'})
        elif "capture" in command:
            ret, frame = cap.read() if cap else (True, np.zeros((240, 320, 3), dtype=np.uint8))
            if ret:
                description, response = analyze_image(frame)
                return jsonify({'command': 'capture', 'description': description, 'response': response})
        return jsonify({'command': None})
    except queue.Empty:
        return jsonify({'command': None})

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
    finally:
        if cap:
            cap.release()
