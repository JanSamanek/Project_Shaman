import subprocess

print(f"[INF] Deploying Gstreamer pipeline ...")
host = "192.168.0.159"
gstreamer_port = 5000
pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={host} port={gstreamer_port}"
print(pipeline)
gstreamer_pipeline = subprocess.Popen(pipeline.split())
print(f"[INF] Streaming video to ip adress: {host}, port: {gstreamer_port} ...")

while True:
    pass