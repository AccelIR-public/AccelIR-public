import math
from PIL import Image

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def concat_image(patch_list, patch_size, o_width, o_height):
    w_space = math.ceil(o_width/patch_size)
    h_space = math.ceil(o_height/patch_size)
    for y in range(h_space):
        for x in range(w_space):
            if x == 0:
                row = patch_list[x + w_space * y]
            else:
                row = get_concat_h(row, patch_list[x + w_space * y])
        if y == 0:
            col = row
        else:
            col = get_concat_v(col, row)
    
    return col