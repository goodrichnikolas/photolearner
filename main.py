import replicate
import os

# Set the API key by importing it from .env file
replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
#export it to the environment
os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

input = {
    "prompt": "A dog wearing a goofy hat that says 'I love Hotdogs'"
}

output = replicate.run(
    "black-forest-labs/flux-schnell",
    input=input
)
print(output)
#=> ["https://replicate.delivery/yhqm/hcDDSNf633zeDUz9sWkKfaf...