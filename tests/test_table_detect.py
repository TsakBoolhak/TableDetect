import pytest
import json
import os
import re
from unittest.mock import patch
import requests
from src.TableDetect import TableDetect, ModelLoadError, ImageLoadError, InvalidImageError, PredictionError

EXCEPTION_MAPPING = {
    "ModelLoadError": ModelLoadError,
    "ImageLoadError": ImageLoadError,
    "InvalidImageError": InvalidImageError,
    "PredictionError": PredictionError
}
TEST_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(TEST_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


@pytest.fixture(scope="module")
def detector():
    """Initialisation of a unique instance of TableDetect for all test
    (except for the TableDetect initialisation failure test of course)"""

    return TableDetect()


def test_model_load_failure():
    """Check if the appropriate exception is raised in case of invalid model given at initialisation"""

    with pytest.raises(ModelLoadError, match=re.escape("Failed to load model.")):
        TableDetect("invalid_model_path")


def test_clear(detector):
    """Check if clear() reset attributes as intended"""

    detector.image_load(os.path.join(TEST_DIR, "images/Bank_statement_1.jpg"))
    detector.predict()
    detector.clear()

    assert detector.image is None, "Expected image to be None after clearing"
    assert detector.results is None, "Expected results to be None after clearing"


def test_image_load_clear_old_results(detector):
    """Check if image_load clears old results"""

    detector.image_load(os.path.join(TEST_DIR, "images/Bank_statement_1.jpg"))
    detector.predict()
    detector.image_load(os.path.join(TEST_DIR, "images/Bank_statement_1.jpg"))

    assert detector.results is None, "Expected results to be None after image_load"


def test_image_load_timeout(detector):
    """Check if image_load raises the correct exception in case of timeout during image download"""
    with patch("requests.get", side_effect=requests.exceptions.Timeout):
        with pytest.raises(ImageLoadError, match=re.escape("Timeout while downloading the image.")):
            detector.image_load("https://trulysmall.com/wp-content/uploads/2023/04/Contractor-Invoice-Template.png",
                                is_url=True)


def test_predict_without_image(detector):
    """Check if calling predict() raises the correct exception if no image were loaded"""

    detector.clear()
    with pytest.raises(ImageLoadError, match=re.escape("No image were loaded. Try using image_load() first.")):
        detector.predict()


def test_print_results_without_image(detector):
    """Check if calling print_results() raises the correct exception if no image were loaded"""

    detector.clear()
    with pytest.raises(ImageLoadError, match=re.escape("No image were loaded. Try using image_load() first.")):
        detector.print_results()


def test_print_results_without_predict(detector):
    """Check if calling print_results() raises the correct exception if predict() were not called"""

    detector.clear()
    detector.image_load(os.path.join(TEST_DIR, "images/Bank_statement_1.jpg"))
    with pytest.raises(PredictionError, match=re.escape("No results available. Try using predict() first.")):
        detector.print_results()


@pytest.mark.parametrize("test_case", config["success_cases"])
def test_table_detection_success(detector, test_case):
    """Testing cases where tables must be successfully detected"""
    if test_case["is_url"] is False:
        image_path = str(os.path.join(TEST_DIR, test_case["image_path"]))
    else:
        image_path = test_case["image_path"]
    detector.image_load(image_path, is_url=test_case["is_url"])

    assert detector.predict() is True, "Expected table(s) to be found"

    assert len(detector.results["boxes"]) == test_case["expected"]["num_boxes"], \
        f"Expected {test_case['expected']['num_boxes']} boxes, but found {len(detector.results['boxes'])}"

    margin = test_case["expected"]["margin"]
    for expected_box, detected_box in zip(test_case["expected"]["boxes"], detector.results["boxes"]):
        detected_box = [round(i, 2) for i in detected_box.tolist()]
        for i in range(4):
            assert abs(expected_box[i] - detected_box[i]) <= margin, \
                f"Box position mismatch: expected {expected_box} but got {detected_box}"


@pytest.mark.parametrize("test_case", config["error_cases"])
def test_table_detection_error(detector, test_case):
    """Testing cases where no table must be found or where exception should be raised"""
    if test_case["is_url"] is False:
        image_path = str(os.path.join(TEST_DIR, test_case["image_path"]))
    else:
        image_path = test_case["image_path"]
    if "exception" in test_case["expected"]:
        expected_exception = EXCEPTION_MAPPING.get(test_case["expected"]["exception"], Exception)
        with pytest.raises(expected_exception, match=re.escape(test_case["expected"]["message"])):
            detector.image_load(image_path, is_url=test_case["is_url"])
    else:
        detector.image_load(image_path, is_url=test_case["is_url"])
        assert detector.predict() is False, "Table(s) found where none should be"
