# exception classes to simplify debugging


class PreprocessingError(Exception):
    pass


class InferenceError(Exception):
    pass


class PostprocessingError(Exception):
    pass


class ImageConversionError(Exception):
    pass
