from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import pathlib
from PIL import Image

# Get endpoint and key from environment variables
import os
endpoint = 'your cognitive service endpoint here'
key = 'your key here'

# Set credentials
credentials = CognitiveServicesCredentials(key)

# Create client
client = ComputerVisionClient(endpoint, credentials)

print("Done!")

path = "path/url to analyze"

# url = pathlib.Path(path1).as_uri()
print("URI+>"+path)
image = open(path, 'rb')

print("#-----------------IMAGE ANALYSIS-------------#")
#-----------------IMAGE ANALYSIS-------------#
image_analysis = client.analyze_image_in_stream(image,visual_features=[VisualFeatureTypes.tags])


print(image_analysis)

for tag in image_analysis.tags:
    print(tag)

print("#-----------------DOMAIN ANALYSIS-------------#")
#-----------------DOMAIN ANALYSIS-------------#
models = client.list_models()

for x in models.models_property:
    print(x)


#-----------------DOMAIN ANALYSIS BY PROVIDED NAME-------------#
print("#-----------------DOMAIN ANALYSIS BY PROVIDED NAME-------------#")
# type of prediction
image = open(path, 'rb')
domain = "landmarks"

# Public domain image of Eiffel tower
# url = "https://images.pexels.com/photos/338515/pexels-photo-338515.jpeg"

# English language response
language = "en"

analysis = client.analyze_image_by_domain_in_stream(domain, image, language)

for landmark in analysis.result["landmarks"]:
    print(landmark["name"])
    print(landmark["confidence"])


#-----------------TEXT DESCRIPTION OF IMAGE-------------#
print("#-----------------TEXT DESCRIPTION OF IMAGE-------------#")
image = open(path, 'rb')
domain = "landmarks"
# url = "http://www.public-domain-photos.com/free-stock-photos-4/travel/san-francisco/golden-gate-bridge-in-san-francisco.jpg"
language = "en"
max_descriptions = 3

analysis = client.describe_image_in_stream(image, max_descriptions, language)

for caption in analysis.captions:
    print(caption.text)
    print(caption.confidence)


#-----------------TEXT FROM IMAGE-------------#
print("#-----------------TEXT FROM IMAGE-------------#")
# import models
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
import time
image = open(path, 'rb')

# url = "https://smhttp-ssl-39255.nexcesscdn.net/wp-content/uploads/2016/01/Handwritten-note-on-Presidential-stationery-900x616.jpg"
raw = True
custom_headers = None
numberOfCharsInOperationId = 36

# Async SDK call
rawHttpResponse = client.batch_read_file_in_stream(image, custom_headers,  raw)

# Get ID from returned headers
operationLocation = rawHttpResponse.headers["Operation-Location"]
idLocation = len(operationLocation) - numberOfCharsInOperationId
operationId = operationLocation[idLocation:]

# SDK call
while True:
    result = client.get_read_operation_result(operationId)
    if result.status not in ['NotStarted', 'Running']:
        break
    time.sleep(1)

# Get data
if result.status == TextOperationStatusCodes.succeeded:
    for textResult in result.recognition_results:
        for line in textResult.lines:
            print(line.text)
            print(line.bounding_box)