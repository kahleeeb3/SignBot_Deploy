from flask import Flask, render_template, Response, after_this_request
from typing import Optional, Dict
import threading
import cv2
from werkzeug.serving import make_server
import socket


class _Channel:
    def __init__(self):
        self.latest_jpg: Optional[bytes] = None
        self.frame_id: int = 0
        self.cv = threading.Condition()

class WebServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, jpeg_quality: int = 80):
        
        # Server Params
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.app = Flask(__name__)

        # channels (Video0, Video1)
        self._channels: Dict[str, _Channel] = {
            "video0": _Channel(),
            "video1": _Channel(),
        }

        # Text handling
        self._text = ""
        self._text_version = 0
        self._text_cv = threading.Condition()

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/video0")
        def video0_feed():
            @after_this_request
            def _no_cache(resp):
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp
            return Response(self._gen_mjpeg("video0"),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
        
        @self.app.route("/video1")
        def video1_feed():
            @after_this_request
            def _no_cache(resp):
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp
            return Response(self._gen_mjpeg("video1"),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
        
        # store server and thread
        self._server = None
        self._server_thread: Optional[threading.Thread] = None

        @self.app.route("/text_stream")
        def text_stream():
            @after_this_request
            def _no_cache(resp):
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp

            def gen():
                last = -1
                while True:
                    # wait until text changes or send a keepalive every 15s
                    with self._text_cv:
                        changed = self._text_cv.wait_for(
                            lambda: self._text_version != last, timeout=15.0
                        )
                        if changed:
                            body = self._text
                            last = self._text_version
                            # SSE "message" (one per change)
                            yield f"data: {body}\n\n"
                        else:
                            # keep the connection alive (comment frame)
                            yield ": keep-alive\n\n"

            return Response(gen(), mimetype="text/event-stream")
    
    def start(self):
        # this just starts the server as a thread

        # if thread running, skip
        if self._server_thread and self._server_thread.is_alive():
            return

        # Start the thread (WSGI server to limit terminal output)
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._server_thread.start()

        # print the server address
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Server running at: http://{ip}:{self.port}")
    
    def stream(self, frame, name: str):
        ch = self._channels.get(name)
        if ch is None:
            raise KeyError(f"Unknown channel: {name}")
        
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        jpg = buf.tobytes()

        with ch.cv:
            ch.latest_jpg = jpg
            ch.frame_id += 1
            ch.cv.notify_all()
    
    def _gen_mjpeg(self, name: str):
        ch = self._channels[name]
        last_sent = -1
        while True:
            with ch.cv:
                ch.cv.wait_for(lambda: ch.latest_jpg is not None and ch.frame_id != last_sent, timeout=1.0)
                if ch.latest_jpg is None or ch.frame_id == last_sent:
                    # if no new frame, wait again
                    continue
                jpg = ch.latest_jpg
                last_sent = ch.frame_id

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
    
    def showText(self, html: str):
        # Update text display
        with self._text_cv:
            self._text = html
            self._text_version += 1
            self._text_cv.notify_all()