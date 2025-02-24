from io import BytesIO

import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, UnidentifiedImageError
import logging

DETECTION_MODEL = "TahaDouaji/detr-doc-table-detection"


class TableDetectError(Exception):
    """Base exception for TableDetect errors"""
    pass


class ModelLoadError(TableDetectError):
    """Exception raised during model loading"""
    pass


class ImageLoadError(TableDetectError):
    """Exception raised if image can not be loaded"""
    pass


class InvalidImageError(ImageLoadError):
    """Exception raised if image file is not valid"""
    pass


class PredictionError(TableDetectError):
    """Exception raised during prediction"""
    pass


class TableDetect:
    def __init__(self, model_path: str = DETECTION_MODEL):
        self._processor = None
        self._model = None
        self._image = None
        self._results = None

        self.model_load(model_path)

    @property
    def results(self):
        return self._results

    @property
    def image(self):
        return self._image

    def model_load(self, modelPath: str = DETECTION_MODEL) -> None:
        """Table detection processor and model loading"""

        try:
            self._processor = DetrImageProcessor.from_pretrained(modelPath)
            self._model = DetrForObjectDetection.from_pretrained(modelPath)

        except OSError as e:
            raise ModelLoadError("Failed to load model.") from e

        except Exception as e:
            raise ModelLoadError("Unexpected error during model loading.") from e

    def image_load(self, path: str, is_url: bool = False) -> None:
        """Load an image from local file or URL"""

        try:
            if self._image is not None:
                self.clear()
            if is_url:
                response = requests.get(path, timeout=15)
                response.raise_for_status()
                self._image = Image.open(BytesIO(response.content))
            else:
                self._image = Image.open(path)

            self._image = self._image.convert("RGB")

        except requests.exceptions.Timeout:
            raise ImageLoadError("Timeout while downloading the image.")

        except requests.exceptions.RequestException as e:
            raise ImageLoadError("Failed to download image.") from e

        except UnidentifiedImageError:
            raise InvalidImageError("The provided file is not a valid image.")

        except FileNotFoundError:
            raise ImageLoadError("Image file not found.")

        except Exception as e:
            raise ImageLoadError("Unexpected error during image loading.") from e

    def clear(self) -> None:
        """clear image and tied results"""

        if self._image is not None:
            self._image.close()
            self._image = None
            self._results = None

    def predict(self, threshold: float = 0.9) -> bool:
        """Use model to predict if tables were found in the loaded image.
        Store the results in results attribute."""

        if self._image is None:
            raise ImageLoadError("No image were loaded. Try using image_load() first.")

        if threshold < 0 or threshold > 1:
            raise PredictionError("Threshold must be between 0 and 1.")

        try:
            inputs = self._processor(images=self._image,
                                     return_tensors="pt")
            outputs = self._model(**inputs)

            target_sizes = torch.tensor([self._image.size[::-1]])
            self._results = self._processor.post_process_object_detection(outputs,
                                                                          target_sizes=target_sizes,
                                                                          threshold=threshold)[0]
            if len(self._results["scores"]) == 0:
                return False
            return True

        except Exception as e:
            raise PredictionError("An unexpected error occurred during prediction.") from e

    def print_results(self) -> None:
        """prints in the console informations about the predicted tables found"""

        if self._image is None:
            raise ImageLoadError("No image were loaded. Try using image_load() first.")

        if self._results is None:
            raise PredictionError("No results available. Try using predict() first.")

        if len(self._results["scores"]) == 0:
            print("No table found in the image.")
        else:
            for score, label, box in zip(self._results["scores"], self._results["labels"], self._results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {self._model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

    def generate_table_images(self) -> list[Image]:
        """Extract table areas from the image and return them as a list of PIL images"""

        if self._image is None:
            raise ImageLoadError("No image was loaded. Try using image_load() first.")

        if self._results is None:
            raise PredictionError("No results available. Try using predict() first.")

        table_images = []
        for i, box in enumerate(self._results["boxes"]):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            try:
                cropped_image = self._image.crop((x_min, y_min, x_max, y_max))
                if cropped_image is None:
                    raise ValueError(f"Image cropping failed for table #{i+1}.")
                table_images.append(cropped_image)

            except ValueError as e:
                logging.error(f"Error processing table {i+1}: {str(e)}")
                continue

            except OSError as e:
                logging.error(f"Error processing table {i+1}: {str(e)}")
                continue

            except Exception as e:
                logging.error(f"Unexpected error processing table {i + 1}: {str(e)}")
                continue

        return table_images
