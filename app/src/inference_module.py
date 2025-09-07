import numpy as np
import onnxruntime as rt
from pathlib import Path
from config import config


# onnxrt module for inference
class InferenceModule:
    def __init__(self, model: str):
        rt.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)
        self._model_name = model
        self._model_path = config.MODEL_PATHS[model]
        self._model_name = model
        self._exec_providers = config.EXEC_PROVIDERS
        self._session = rt.InferenceSession(
            config.MODEL_PATHS[model], providers=self._exec_providers
        )
        if config.ENABLE_FALLBACK:
            self._session.enable_fallback()
        self._input_name = self._session.get_inputs()[0].name

    def change_model(self, model: str):
        self._session = rt.InferenceSession(
            config.MODEL_PATHS[model], providers=self._exec_providers
        )
        self._session.enable_fallback()
        self._model_name = model

    def get_current_model_name(self):
        return self._model_name

    def infer_single_batch(self, batch: np.ndarray):
        if len(batch.shape) != 4:
            raise TypeError(
                "The method expects a single batch of RGB images as an argument"
            )

        output = self._session.run([], {self._input_name: batch})[0]

        return output

    def infer_batches(self, inputs_batched: list):
        if not isinstance(inputs_batched, list):
            raise TypeError(
                "The method expects a list of four-dimensional batches of RGB as an argument"
            )

        outputs = [
            self._session.run([], {self._input_name: inputs_batched[i]})[0]
            for i in range(len(inputs_batched))
        ]

        return outputs
