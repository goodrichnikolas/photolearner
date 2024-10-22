import spacy
from PIL import Image as PILImage
from PIL import ImageFont, ImageDraw

# Load the Spanish language model
# nlp = spacy.load("es_core_news_sm")

class Vocabulary:
    def __init__(self, untransformed) -> None:
        self.untransformed = untransformed

    def get_infinitive(self, verb):
        """
        Returns the infinitive form of a given verb.
        """
        doc = nlp(verb)
        return doc[0].lemma_

class Flux:
    def __init__(self, prompt) -> None:
        self.prompt = prompt

class Image:
    def __init__(self) -> None:
        self.image_path = None

    def open_image(self, image_path):
        self.image_path = image_path

    def add_vocab_to_image(self, image_path, vocab):
        """
        Loads the image and adds a black box with white text centered
        at the bottom of the image with the vocabulary word.
        """
        # Open the image using PIL
        image = PILImage.open(image_path)
        
        # Get the dimensions of the image
        width, height = image.size
        
        # Create a new image with the same dimensions plus extra space for text
        new_image = PILImage.new('RGB', (width, height + 50), color = (0, 0, 0))
        
        # Paste the original image onto the new image
        new_image.paste(image)
        
        # Create an ImageDraw object
        draw = ImageDraw.Draw(new_image)
        
        # Calculate the font size proportional to the image width
        font_size = int(width * 0.05)  # 5% of the image width
        
        # Load a font with the calculated size
        font = ImageFont.truetype("times.ttf", font_size)
        
        # Get the size of the text using textbbox
        text = vocab
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        
        # Calculate the position to center the text
        text_x = (width - text_width) / 2
        text_y = height + (50 - text_height) / 2
        
        # Draw the text
        draw.text((text_x, text_y), text, (255, 255, 255), font=font)
        
        return new_image