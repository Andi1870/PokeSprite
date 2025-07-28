from PIL import Image

img = Image.open(r"data\Generation_2\abra_Normal_Generation_2.png")  # z. B. "sprites/pikachu.png"
img.show()
print(img.getpixel((0, 0))) 

img = Image.open(r"data\Generation_3\abra_Normal_Generation_3.png")  # z. B. "sprites/pikachu.png"
img.show()
print(img.getpixel((0, 0))) 

img = Image.open(r"data\Generation_4\abra_Normal_Generation_4.png")  # z. B. "sprites/pikachu.png"
img.show()
print(img.getpixel((0, 0))) 

img = Image.open(r"data\Generation_5\abra_Normal_Generation_5.png")  # z. B. "sprites/pikachu.png"
img.show()
print(img.getpixel((0, 0))) 