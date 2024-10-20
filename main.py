import os
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import time
from vocab import Vocabulary

# Set the API key by importing it from .env file
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
# Export it to the environment
os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

def save_base64_image(base64_string, output_path):
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Open the image using PIL
    image = Image.open(BytesIO(image_data))
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")






def main():
    # Define the input data
    input_data = {
        "input": {
            "prompt": "A picture of a banana split ",
        }
    }

    # Make the POST request to the Replicate API
    response = requests.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        headers={
            "Authorization": f"Bearer {replicate_api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait"
        },
        data=json.dumps(input_data)
    )

    # Check if the request was successful
    if response.status_code == 200 or response.status_code == 201:
        response_json = response.json()
        print("Full API response:", response_json)

        # Check if the response contains an 'output' field with image data
        if 'output' in response_json and isinstance(response_json['output'], list) and len(response_json['output']) > 0:
            base64_image = response_json['output'][0]
            
            # Create a 'downloads' directory if it doesn't exist
            os.makedirs('downloads', exist_ok=True)
            
            # Save the image
            output_path = os.path.join('downloads', f"output_{int(time.time())}.jpg")
            save_base64_image(base64_image, output_path)
        else:
            print("No image data found in the response")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()