import os
import requests
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import time
from classes import Vocabulary, Flux, Image
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key by importing it from .env file
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if not replicate_api_token:
    raise EnvironmentError("REPLICATE_API_TOKEN is not set in the environment.")

def save_base64_image(base64_string, output_path):
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Open the image using PIL
    image = PILImage.open(BytesIO(image_data))
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

def add_vocab_to_image(image_path, vocab_word):
    # Read the image
    img = plt.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Remove axis
    ax.axis('off')

    # Add text
    ax.text(0.5, 0.95, vocab_word, 
            fontsize=40, color='white', 
            horizontalalignment='center', 
            verticalalignment='top', 
            transform=ax.transAxes,
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=5))

    # Save the figure
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return PILImage.open(image_path)

def main():
    # Define the input data
    example_word = sys.argv[1]

    # Create a Vocabulary object
    vocabulary = Vocabulary(sys.argv[1])

    # Create an Image object
    image = Image()

    # Create a Flux object
    flux = Flux(sys.argv[2])

    input_data = {
        "input": {
            "prompt": flux.prompt,
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
            output_path = os.path.join('downloads', f"{vocabulary.untransformed}.png")
            save_base64_image(base64_image, output_path)
            image.open_image(output_path)
        else:
            print("No image data found in the response")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

    # Add the vocabulary word to the image and save it over the original image
    new_image = add_vocab_to_image(image.image_path, vocabulary.untransformed)
    new_image.save(image.image_path)
    print("Vocabulary added to the image")

    print("Process completed successfully")

if __name__ == "__main__":
    main()