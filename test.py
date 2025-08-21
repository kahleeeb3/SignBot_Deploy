from flask import Flask, Response, render_template
import time
import json
import threading
import queue

app = Flask(__name__)
message_queue = queue.Queue()

def update_display_text(new_text):
    message_queue.put(json.dumps({'message': new_text}))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/stream')
def stream():
    # Sends messages from the server to the client as they become available.
    def event_stream():
        while True:
            # Wait until a message is available in the queue
            data = message_queue.get()
            yield f'data: {data}\n\n'

    return Response(event_stream(), mimetype="text/event-stream")

def simulated_updates():
    count = 0
    while True:
        count += 1
        new_text = f"The server updated this text. (Update #{count})"
        update_display_text(new_text)
        time.sleep(5)  # Wait 5 seconds before the next update


if __name__ == '__main__':

    update_thread = threading.Thread(target=simulated_updates, daemon=True)
    update_thread.start()
    
    app.run(host="0.0.0.0", port=8765, debug=False, threaded=True)