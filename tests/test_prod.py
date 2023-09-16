import io
import os
from glob import glob

import decouple
import PIL.Image as Image
import requests

# Get the URL and the endpoint
URL = decouple.config("URL_PROD")
ENDPOINT = "eq"
ENDPOINT_URL = os.path.join(URL, ENDPOINT)

# Get the files
files = glob("files/*.mp3") + glob("files/.wav")

# Create dummy tracks
tracks = [f"track_{i}" for i in range(len(files))]

# Create the files to be opened and passed to the model
files = [
    ("audio_files", (track, open(file, "rb"), "audio/mpeg"))
    for track, file in zip(tracks, files)
]

# Make the request
response = requests.request("POST", ENDPOINT_URL, files=files)

# Path where to save the corresponding graph
save_path = "graph.png"

# If call was successful, retrieve the image
if response.status_code == 200:
    # Retrieve the image bytes
    image_bytes = response.content

    # Save image locally
    Image.open(io.BytesIO(image_bytes)).save(save_path)

else:
    # Print the logs of the errors
    print(response.status_code)
    print(response.text)
