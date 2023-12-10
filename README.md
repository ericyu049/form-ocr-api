# Form OCR API

This is the backend server for extracting text from an uploaded image. It provides a REST API for uploading the image and returning the extracted text.
This is a demo using a model trained with dataset downloaded from [here](https://www.nist.gov/srd/nist-special-database-2), and the model training details can be found in this [repo](https://github.com/ericyu049/form-ocr).




## Usage

### Run
Run ```python3 docupload.py```

### Endpoints

The endpoint takes an image as form data. 

POST http://localhost:8080/upload

photo: your image byte


### Demo

For demo project please checkout this [repo](https://github.com/ericyu049/form-ocr-ui)
