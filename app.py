import time
import edgeiq
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from collections import deque

app = Flask(__name__, template_folder='./templates/')

socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(
    app, logger=socketio_logger, engineio_logger=socketio_logger)

SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.FileVideoStream("fire0.m4v", play_realtime=True)
obj_detect = edgeiq.ObjectDetection("dvarney/daily_alligator")
obj_detect.load(engine=edgeiq.Engine.DNN)
SAMPLE_RATE = 50


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect')
def connect_cv():
    print('[INFO] connected: {}'.format(request.sid))


@socketio.on('disconnect')
def disconnect_cv():
    print('[INFO] disconnected: {}'.format(request.sid))


@socketio.on('close_app')
def close_app():
    print('Stop Signal Received')
    controller.close()


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler
        (https://github.com/alwaysai/video-streamer) and is modified here.

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.all_frames = deque()
        self.video_frames = deque()
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        # loop detection
        video_stream.start()

        socketio.sleep(0.01)
        self.fps.start()

        # loop detection
        while True:
            try:
                frame = video_stream.read()
                text = [""]
                socketio.sleep(0.02)

                # run CV here
                results = obj_detect.detect_objects(
                    frame, confidence_level=.1, overlap_threshold=0.3)

                predictions = results.predictions
                frame = edgeiq.markup_image(frame, predictions, colors=obj_detect.colors, line_thickness=2, font_size=0.7, show_confidences = False)
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Frame Rate: {:1f} FPS".format(1/results.duration))
                text.append("Objects:")

                for prediction in predictions:
                    text.append("{}".format(
                        prediction.label))
                self.send_data(frame, text)
                socketio.sleep(0.01)
                self.fps.update()

                if self.check_exit():
                    video_stream.stop()
                    controller.close()
            except edgeiq.NoMoreFrames:
                video_stream.start()

    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=720, height=480, keep_scale=True)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)
                    })
            socketio.sleep(0.01)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.exit_event.is_set()

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()


class Controller(object):
    def __init__(self):
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('[INFO] Starting server at http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.fps.stop()
        print("elapsed time: {:.2f}".format(self.fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(self.fps.compute_fps()))

        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()

        print("Program Ending")


controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        controller.close()
