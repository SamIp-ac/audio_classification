import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ic_model import instruments_classify as ic
import os
from PIL import Image
import urllib.request
from IPython.display import Audio
import librosa
import io
from six.moves.urllib.request import urlopen
from uvicorn import run

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/")
async def root():
    return {"message": "Welcome to the mxl to instruments classification API with pydantic!!!"}


@app.post("/net/inst_classify")
async def inst_classify(url):
    ans = ic.inst_classifier(url=url)
    prediction = ic.predict_inst(ans)

    return {'audio prediction': ans, 'instrument prediction': prediction}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)

