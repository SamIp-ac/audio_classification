import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ic_model import instruments_classify as ic
import os
from uvicorn import run
from pydantic import BaseModel, Field
from typing import Optional, List


class inst_pred(BaseModel):
    url: Optional[str]
    inst_pred: Optional[str]

    class Config:
        orm_mode = True
        extra = "allow"


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


@app.post("/net/inst_classify/")
async def inst_classify(param: inst_pred):
    param.audio_pred = ic.inst_classifier(url=param.url)
    param.inst_pred = ic.predict_inst(param.audio_pred)

    return {'audio prediction': param.audio_pred, 'instrument prediction': param.inst_pred}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
