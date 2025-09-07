from inference_module import InferenceModule
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from image_processing import PREPROCESSING_PIPELINE, POSTPROCESSING_PIPELINE
from config import config
import base64
import zipfile
import tempfile
from typing import Optional, Literal
from pydantic import BaseModel, field_validator
from exceptions import (
    PreprocessingError,
    InferenceError,
    PostprocessingError,
    ImageConversionError,
)


app = FastAPI()
inf_module = InferenceModule(model=config.AVAILABLE_MODELS[0])


# predict request model
class PredictRequest(BaseModel):
    model: str
    format_as: Literal[".png", ".jpg", ".tif"]
    threshold: Optional[float] = 0
    get_overlayed: bool = False
    as_json: bool = False
    filename: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in config.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{v}' is not supported. Available: {config.AVAILABLE_MODELS}"
            )
        return v

    @field_validator("threshold", mode="before")
    @classmethod
    def validate_threshold(cls, v):
        if isinstance(v, float):
            v = v if v > 0 else None


@app.get("/health", tags=["Check health"])
async def health():
    return {"status": "ok!"}


@app.post(
    "/segment_buildings",
    description="Segment buildings on aerial satellite images",
    tags=["Predict"],
)
async def predict(
    file: UploadFile = File(..., description="Source image"),
    model=Form(..., description="Model", enum=config.AVAILABLE_MODELS),
    format_as: Literal[".png", ".jpg", ".tif"] = Form(..., description="Target format"),
    threshold: Optional[float] = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Threshold to form a segmentation mask. Ignore to only get a heatmap",
    ),
    get_overlayed: bool = Form(
        False,
        description="True to get the source image overlayed with the mask or the heatmap along with the result",
    ),
    as_json: bool = Form(
        False, description="True to get results as a JSON base64-encoded file"
    ),
    filename: Optional[str] = Form(
        None,
        description="Filename for the result. Ignore to leave the same as the source",
    ),
):
    """
    Executes end-to-end builing segmentation pipeline and returns json / zip archive of resulting images.
    """
    PredictRequest(
        model=model,
        format_as=format_as,
        threshold=threshold,
        as_json=as_json,
        get_overlayed=get_overlayed,
        filename=filename,
    )

    try:
        if inf_module.get_current_model_name() != model:
            inf_module.change_model(model)

        # get current model config
        current_config = config.MODEL_CONFIGS[inf_module.get_current_model_name()]
        contents = await file.read()

        # preprocess an image depending on the current model
        try:
            data, meta = PREPROCESSING_PIPELINE[model](
                img=contents, config=current_config, get_input_img=get_overlayed
            )
        except Exception as e:
            raise PreprocessingError(f"Image prepocessing error. Trace: {str(e)}")

        # infer data
        try:
            if isinstance(data, list):
                outputs = inf_module.infer_batches(data)
            else:
                outputs = inf_module.infer_single_batch(data)
        except Exception as e:
            raise InferenceError(f"Inference Error. Trace: {str(e)}")

        # postprocess an image depending on the current model
        try:
            data = POSTPROCESSING_PIPELINE[model](
                data=outputs,
                meta=meta,
                config=current_config,
                format_as=format_as,
                threshold=threshold,
                get_overlayed=get_overlayed,
            )
        except Exception as e:
            raise PostprocessingError(f"Image postprocessing error. Trace: {str(e)}")

        # prepare data for response
        try:
            if as_json:
                for k, v in data.items():
                    data[k] = base64.b64encode(v.tobytes()).decode("utf-8")
                return data
            else:
                zip_file = make_zip(data, format_as)
                filename = (
                    file.filename.split(".")[0] if filename == "string" else filename
                )
                return FileResponse(
                    zip_file, filename=filename, media_type="application/zip"
                )
        except Exception as e:
            raise ImageConversionError(f"Image conversion error. Trace: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def make_zip(images: dict[str, bytes], format_as: str):
    """Create a zip archive of images.

    Args:
        images (dict[str, bytes]): Encoded images.
        format_as (str): Format to convert into.

    Returns:
        _type_: Buffer of archive.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp, mode="w") as zf:
        for name, img_bytes in images.items():
            zf.writestr(f"{name}{format_as}", img_bytes)
    tmp.close()
    return tmp.name
