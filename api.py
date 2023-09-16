import modal
from modal import asgi_app, stub

# Initialise Modal stub (necessary for using Modal)
stub = modal.Stub("audio-graph-api")

# Create image (Docker-like) to be used by Modal backend
image = modal.Image.debian_slim(python_version="3.10")

# Pip install packages
image = image.pip_install("librosa", "polars", "matplotlib")

# Assign image to stub
stub.image = image


@stub.function(image=image)
@asgi_app()
def app():
    from fastapi import FastAPI

    # Import endpoints to be deployed
    from src import audio_graph

    # Initialise FastAPI app
    app = FastAPI()

    # Include necessary endpoints
    app.include_router(audio_graph.router)

    return app
