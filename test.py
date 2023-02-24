import subprocess

pipeline = "gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host=192.168.88.82 port=5000"

process = subprocess.Popen(pipeline, stdout=subprocess.PIPE, shell=True)

# Read the output
output, errors = process.communicate()

# Print the output
print(output.decode())

# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink
# gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host=192.168.0.159 port=5000

# git push https://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@github.com/JanSamanek/Project_Shaman.git
# token: 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