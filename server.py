from flask import Flask, render_template, Response
from typing import Optional
import threading
import cv2
from werkzeug.serving import make_server


class WebServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, jpeg_quality: int = 80):
        
        # Server Params
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.app = Flask(__name__)

        # Frame Storage
        self._latest_jpg: Optional[bytes] = None
        self._frame_id = 0

        # Threading (This will be a subprocess)
        self._cv = threading.Condition()
        self._server_thread: Optional[threading.Thread] = None

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/video0")
        def video_feed():
            return Response(self._gen_mjpeg(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
    
    def start(self):
        # this just starts the server as a thread

        # if thread running, skip
        if self._server_thread and self._server_thread.is_alive():
            return

        # Start the thread (WSGI server to limit terminal output)
        self._server = make_server(self.host, self.port, self.app)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._server_thread.start()

        # print the server address
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Server running at: http://{ip}:{self.port}")
    
    def stream(self, frame_bgr):
        # encode frame as jpeg and inform thread of new frame arrival
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        jpg = buf.tobytes()
        with self._cv:
            self._latest_jpg = jpg
            self._frame_id += 1
            self._cv.notify_all()
    
    def _gen_mjpeg(self):
        # the /video0 route will continuously call this.
        # it will wait until a new frame is passed
        last_sent = -1
        while True:
            with self._cv:
                # Wait until a new frame arrives
                self._cv.wait_for(lambda: self._latest_jpg is not None and self._frame_id != last_sent)
                jpg = self._latest_jpg
                last_sent = self._frame_id

            # Yield one MJPEG part
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
            
"""
server = WebServer(port=8765, jpeg_quality=80)
server.start()

camera = FrameCapture()
camera.open()

while True:
    camera.capture()
    if camera.frame is None:
        break
    
    # cv2.imshow("frame", camera.frame)
    server.stream(camera.frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
"""