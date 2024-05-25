import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


image = Image.open('C:/Users/abdul/Downloads/in.jpeg')

text = pytesseract.image_to_string(image)


print(text)


# download tesseract-ocr-w64-setup-5.3.3.20231005.exe for windows 64-bit
