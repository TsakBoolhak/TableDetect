# TableDetect - Table Detection with Transformers

![Tests Status](https://github.com/TsakBoolhak/TableDetect/actions/workflows/tests.yml/badge.svg)

TableDetect is a Python class for table detection in images using the `TahaDouaji/detr-doc-table-detection` pre-trained model to detect tables in bank statements, invoices, and other document images.

## Features

- **Model Loading**: Load the pre-trained table detection model.
- **Image Loading**: Load images from file or URL.
- **Prediction**: Detect tables in the image and provide their bounding boxes.
- **Error Handling**: Handle various error scenarios (e.g., invalid images, timeouts).
- **Test Suite**: Includes a comprehensive set of tests with pytest for table detection success and error cases.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/TsakBoolhak/TableDetect.git TableDetect
   ```
2. Navigate to the project directory:
    ```bash
   cd TableDetect
    ```
3. Install the required dependencies: 
```bash
pip install -r requirements.txt
```

## Usage

### TableDetect Class
The main class for table detection is `TableDetect`. Here is an example of how to use it:

```python
from src.TableDetect import TableDetect

# Initialize the TableDetect class
detector = TableDetect()

# You also can initialize a TableDetect with another model,
# or load it into an already instantiated TableDetect
detector.model_load("path/to/alternative/model")
detector2 = TableDetect("path/to/alternative/model")

# Load an image from a file
detector.image_load("path_to_image.jpg")

# Load an image from a URL
detector.image_load("https//path/to/url.png", is_url=True)

# Perform table detection with the default threshold (only keeps results equal or above 90% confidence)
# or set the desired threshold (value from 0 to 1)
detector.predict()
detector2.predict(threshold=0.4)

# You can access the results via the results attribute / getter
results = detector.results
results_2 = detector2.results()

# Print information about the results
detector.print_results()

# Once predict() is used you also can retrieve a list of PIL Images
# that are cropped images of the areas of the document where tables were found
table_images = detector.generate_table_images()
for i, table in enumerate(table_images):
   table.show(title=f"Table {i+1}")
```

### Methods

- `model_load(model_path: str = "TahaDouaji/detr-doc-table-detection")`: Load the table detection model, model_path is an optional argument to specify a path to a detection model, its default value is the model trained by TahaDouaji.
- `image_load(path: str, is_url: bool = False)`: Load an image either from a local file or a URL (if that is the case you should set the is_url optional parameter to True).
- `clear()`: Clears the loaded image and the tied detection results.
- `predict(threshold: float = 0.9)`: Detect tables in the loaded image, returns True if tables are detected, False otherwise. In both cases the results attribute is set to store the results (empty list in case of no detection).
- `print_results()`: display the detected tables and their bounding boxes in the console.
- `generate_table_images()`: Returns a list of cropped table images from the original document.
- `results()`: getter for the _results attribute, can also be accessed as an attribute (`detector.results`).
- `image()`: getter for the _image attribute, can also be accessed as an attribute (`detector.image`).

### Error Handling

The following exceptions may be raised:
- `ModelLoadError`: Raised if the model fails to load.
- `ImageLoadError`: Raised if there are any issues loading an image (timeout, invalid path, invalid URL).
- `InvalidImageError`: Raised in case the provided image is corrupted or in an unhandled format.
- `PredictionError`: Raised in case of an error occurring during prediction.

## Running Tests
To run the tests for the project, simply use the following command:
```bash
pytest
```

## Test Cases
The test suite covers both successful and error scenarios, including:
- Successful table detections with validation of the detection boxes.
- Error handling for invalid images, timeout, model loading issues and various misuses of the class.

## Adding Tests

To add new tests you just have to add a test case in the `config.json` file located in the `tests` directory.
You also can edit `test_table_detect.py` file in the same directory to add more specific tests

Example of adding a test case in `config.json`:
```json
{
   "success_cases": [
      {
         "image_path": "new_image.jpg",
         "is_url": false,
         "expected": {
            "num_boxes": 1,
            "boxes": [
               [100, 200, 500, 800]
            ],
            "margin": 10
         }
      },
      {
         ...
      }
   ]
   "error_cases": [
      {
         "image_path": "images/no_table.jpg", 
         "is_url": false, 
         "expected": {
           "table_found": false
         }
      },
      {
         "image_path": "images/empty.jpg", 
         "is_url": false, 
         "expected": {
            "exception": "InvalidImageError", 
            "message": "The provided file is not a valid image."
         }
      },
      {
         ...
      }
   ]
}
```
## CI/CD with GitHub Actions
The repository includes a GitHub Actions workflow (`.github/workflows/tests.yml`) that automatically runs the tests on every push or pull request to the `master` branches.
### Workflow Summary
- **Install Dependencies**: Installs the required dependencies from requirements.txt
- **Run Test**: Runs the tests using `pytest`.
- **Manual Trigger**: You can manually trigger the workflow via the GitHub Actions UI.

To trigger the workflow manually, navigate to the "Actions" tab on GitHub, select the workflow (**Run Tests**) and click "Run workflow".
