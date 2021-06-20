
def crop_center(img, cropx=700, cropy=700):
    y, x = img.shape
    low = min(x, y)
    if low < 700:
        print("UNDER 700 image crop")
        cropx = low
        cropy = low

    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]
