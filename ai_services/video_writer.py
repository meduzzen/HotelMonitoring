import subprocess
import threading
import queue
import cv2
import numpy as np


class VideoOutput:
    """Handles writing frames to a file or RTSP stream in a background thread."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: str | None = None,
        stream_url: str | None = None,
    ):
        self.stream = False
        self.running = True

        self.write_queue = queue.Queue(maxsize=int(fps * 2))

        if output_path:
            self.writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        elif stream_url:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-f",
                "rtsp",
                stream_url,
            ]
            self.writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            self.stream = True
        else:
            raise ValueError("Either output_path or stream_url must be provided.")

        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        """Continuously pulls frames from the queue and writes them."""
        while self.running or not self.write_queue.empty():
            try:
                frame = self.write_queue.get(timeout=0.1)

                if frame is None:
                    break

                if self.stream:
                    self.writer.stdin.write(frame.tobytes())
                else:
                    self.writer.write(frame)

            except queue.Empty:
                continue
            except BrokenPipeError:
                print("FFmpeg pipe broken. Stream might be down.")
                break
            except Exception as e:
                print(f"Error writing frame: {e}")

    def write(self, frame: np.ndarray):
        """Adds a frame to the queue. Drops it if the queue is full."""
        if not self.running:
            return

        try:
            self.write_queue.put(frame.copy(), block=False)
        except queue.Full:
            pass

    def release(self):
        """Cleanly shuts down the writer thread and releases resources."""
        self.running = False

        try:
            self.write_queue.put(None, block=False)
        except queue.Full:
            pass

        if self.thread.is_alive():
            self.thread.join(timeout=3.0)

        if self.stream:
            if self.writer.stdin:
                self.writer.stdin.close()
            self.writer.wait()
        else:
            self.writer.release()
