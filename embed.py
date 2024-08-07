import cv2
import numpy as np
import math
import skfuzzy as fuzz
from PIL import Image
import os
from collections import deque

class qgdec:

    def __init__(self, im, bits):
        self.img = im
        self.bits = bits

    def createblock(self):
        self.Embed_lvl_queue = []
        new_block = []
        # If the image is not even, add an extra row or column
        if self.img.shape[0] % 2 == 1:
            self.img = np.vstack([self.img, self.img[-1, :]])  # Add an extra row with same PV as last row
        if self.img.shape[1] % 2 == 1:
            self.img = np.hstack([self.img, self.img[:, -1].reshape(-1, 1)])  # Add an extra column with same PV as last column
        
        # Iterate over each pixel in the image
        for i in range(0, self.img.shape[0], 2):
            for j in range(0, self.img.shape[1], 2):
                # Append the pixel value to the list
                block = [self.img[i, j], self.img[i, j + 1], self.img[i + 1, j], self.img[i + 1, j + 1]]
                # Fuzzy difference
                h = [
                    abs(self.img[i, j] - self.img[i, j + 1]), 
                    abs(self.img[i, j + 1] - self.img[i + 1, j]), 
                    abs(self.img[i, j] - self.img[i + 1, j]), 
                    abs(self.img[i, j] - self.img[i + 1, j + 1]), 
                    abs(self.img[i + 1, j] - self.img[i + 1, j + 1]), 
                    abs(self.img[i, j + 1] - self.img[i + 1, j + 1])
                ]
                h_avg = np.mean(h)
                h_med = np.median(h)
                Embed_lvl = np.zeros_like(self.img, dtype=float)  # Ensure Embed_lvl is a float array
                mean_value = np.mean([self.embedlvl(h_avg), self.embedlvl(h_med)])
                Embed_lvl[i, j] = mean_value
                # self.Embed_lvl_queue.append(Embed_lvl[i, j])
                avg_pixblock = np.mean(block)
                if avg_pixblock <= 150:
                    kb1 = self.img[i, j] - (self.img[i, j] % 4)
                    kb2 = self.img[i, j + 1] - (self.img[i, j + 1] % 4)
                    kb3 = self.img[i + 1, j] - (self.img[i + 1, j] % 4)
                    kb4 = self.img[i + 1, j + 1] - (self.img[i + 1, j + 1] % 4) 
                    d1 = self.eq8(self.img[i, j], kb1)
                    d2 = self.eq8(self.img[i, j+1], kb2)
                    d3 = self.eq8(self.img[i+1, j], kb3)
                    d4 = self.eq8(self.img[i+1, j+1], kb4)
                    dnew1 = self.eq10(d1)
                    dnew2 = self.eq10(d2)
                    dnew3 = self.eq10(d3)
                    dnew4 = self.eq10(d4)
                    dnewnew1 = self.eq11(dnew1, Embed_lvl[i,j])
                    dnewnew2 = self.eq11(dnew2, Embed_lvl[i,j])
                    dnewnew3 = self.eq11(dnew3, Embed_lvl[i,j])
                    dnewnew4 = self.eq11(dnew4, Embed_lvl[i,j])
                    new_px1 = self.eq12(dnewnew1, kb1)
                    new_px2 = self.eq12(dnewnew2, kb2)
                    new_px3 = self.eq12(dnewnew3, kb3)
                    new_px4 = self.eq12(dnewnew4, kb4)
                else:
                    ka1 = self.img[i, j] + abs((self.img[i, j] % 4) - 3)
                    ka2 = self.img[i, j + 1] + abs((self.img[i, j + 1] % 4) - 3)
                    ka3 = self.img[i + 1, j] + abs((self.img[i + 1, j] % 4) - 3)
                    ka4 = self.img[i + 1, j + 1] + abs((self.img[i + 1, j + 1] % 4) - 3)
                    d1 = self.eq8(self.img[i, j], ka1)
                    d2 = self.eq8(self.img[i, j+1], ka2)
                    d3 = self.eq8(self.img[i+1, j], ka3)
                    d4 = self.eq8(self.img[i+1, j+1], ka4)
                    dnew1 = self.eq10(d1)
                    dnew2 = self.eq10(d2)
                    dnew3 = self.eq10(d3)
                    dnew4 = self.eq10(d4)
                    dnewnew1 = self.eq11(dnew1, Embed_lvl[i,j])
                    dnewnew2 = self.eq11(dnew2, Embed_lvl[i,j])
                    dnewnew3 = self.eq11(dnew3, Embed_lvl[i,j])
                    dnewnew4 = self.eq11(dnew4, Embed_lvl[i,j]) 
                    new_px1 = self.eq13(dnewnew1, ka1)
                    new_px2 = self.eq13(dnewnew2, ka2)
                    new_px3 = self.eq13(dnewnew3, ka3)
                    new_px4 = self.eq13(dnewnew4, ka4)
                block_pxlval = np.array([new_px1, new_px2, new_px3, new_px4])
                new_block.append(block_pxlval) 
        print('block',self.img[2,0])
        print('newblock',new_block[1])
        return new_block
                

    
    def embedlvl(self, characteristics):
        
        categories = {
            'Very small': {'a': -5.0, 'b': -1.0, 'c': 0.0, 'd': 1.0},
            'Small': {'a': 0.5, 'b': 1.0, 'c': 9.5, 'd': 10.5},
            'Small to medium': {'a': 9.5, 'b': 10.5, 'c': 18.0, 'd': 22.0},
            'Medium to large': {'a': 18.0, 'b': 22.0, 'c': 28.0, 'd': 32.0},
            'Large': {'a': 28.0, 'b': 32.0, 'c': 80.0, 'd': 100.0},
            'Very large': {'a': 80.0, 'b': 100.0, 'c': 128.0, 'd': 140.0},
        }
        
        LV = None
        smallest_difference = float('inf')
        
        for category, points in categories.items():
            diff_b = abs(characteristics - points['b'])
            diff_c = abs(characteristics - points['c'])
            temp_smallest_diff = min(diff_b, diff_c)
            
            if temp_smallest_diff < smallest_difference:
                smallest_difference = temp_smallest_diff
                LV = category
        
        if LV:
            selected_points = categories[LV]
            
        trapezoid = max(min((characteristics - selected_points['a']) / (selected_points['b'] - selected_points['a']), 1, 
                            (selected_points['d'] - characteristics) / (selected_points['d'] - selected_points['c'])), 0)

        categories = {
        'Very small': {'a': -1.0,'b': 0, 'c': 0.25, 'd': 0.5},
        'Small': {'a': 0.25,'b': 0.5, 'c': 1.5, 'd': 1.75},
        'Small to moderate': {'a': 1.5,'b': 1.75, 'c': 3.0, 'd': 3.5},
        'Moderate to large': {'a': 3.0,'b': 3.5, 'c': 5.0, 'd': 5.5},
        'Large': {'a': 5.0,'b': 5.5, 'c': 7.0, 'd': 7.5},
        'Very large': {'a': 7.0,'b': 7.75, 'c': 8.0, 'd': 9.0},}
        # Add other categories as needed
        LV = None
        smallest_difference = float('inf')
        for category, points in categories.items():
            diff_b = abs(trapezoid - points['b'])
            diff_c = abs(trapezoid - points['c'])
            temp_smallest_diff = min(diff_b, diff_c)
            # Update the closest category if the current one is closer
            if temp_smallest_diff < smallest_difference:
                smallest_difference = temp_smallest_diff
                LV = category
        if LV:
            selected_points = categories[LV]
        
        x = np.arange(selected_points['a'], selected_points['d'], 0.1)
        mfx = fuzz.trapmf(x, [selected_points['a'], selected_points['b'], selected_points['c'], selected_points['d']])
        Embedlvl = np.ceil(fuzz.centroid(x, mfx))
        return Embedlvl

    @staticmethod
    def eq8(block, kb):
        d = block - kb 
        return d

    @staticmethod
    def eq9(block, ka):
        d = ka - block
        return d

    @staticmethod
    def eq10(d):
        if d <=2:
            dnew =0
        else:
            dnew = d - 2 ** math.floor(math.log2(d))
        return dnew
        
    def eq11(self, dnew, Embed_lvl):

        Embed_lvl = max(0, int(Embed_lvl))

        if Embed_lvl == 0:
            new_bits = 0
        # Check if there are enough bits left
        if len(self.bits) < Embed_lvl:
            # If not, take all remaining bits
            if self.bits:  # Check if self.bits is not empty
                new_bits = int(self.bits, 2)
            else:
                new_bits = 0
            # Set bits to an empty string
            self.bits = ''
        elif len(self.bits) >= Embed_lvl and Embed_lvl > 0:
            if self.bits:
                # If there are enough bits, take the first Embed_lvl bits
                new_bits = int(self.bits[:Embed_lvl], 2)
                # Remove them from bits
                self.bits = self.bits[Embed_lvl:]
            else:
                new_bits = 0
            self.bits = ''
        # Calculate dnewnew
        dnewnew = 2 * dnew + new_bits 

        return dnewnew


    
    @staticmethod
    def eq12(dnewnew, kb):
        stegpx = dnewnew + kb
        return stegpx
    
    @staticmethod
    def eq13(dnewnew, ka):
        stegpx = ka - dnewnew
        return stegpx

    @staticmethod        
    def eq14(self, d):
        # Convert d to binary and remove the '0b' prefix
        binary_d = bin(d)[2:]
        
        # If message is not already a list, initialize it as one
        if not hasattr(self, 'message'):
            self.message = []
        
        # Push the binary value onto the stack
        for bit in binary_d:
            self.message.append(bit)
        
        return self.message

    
    @staticmethod        
    def eq15(d):
        dnew = math.floor(d/2)
        return dnew
    
    @staticmethod
    def eq16(LM, block):
        if LM == '00':
            return 0
        elif LM == '10':
            return block + 2 ^ math.log2(2 * block + 1) + 1
        elif LM == '11':
            return block + 2 ^ math.log2(2 * block + 1)
        else:
            raise ValueError("Invalid LM value. LM should be '00', '10', or '11'.")

    def extracting(block, LM):
        avg_pixblock = np.mean(block)
        if avg_pixblock <= 150:
            kb = [b - (b % 4) for b in block]
            d = qgdec.eq8(block, kb)
            lsb = qgdec.eq14(d)
            dnew= qgdec.eq15(d)
            dnewnew=qgdec.eq16(dnew, LM)
            orgpxl = qgdec.eq8(dnewnew, kb)
        else:
            ka = [b + abs((b % 4) - 3) for b in block]
            d = qgdec.eq9(block)
            lsb = qgdec.eq14(block)
            dnew= qgdec.eq15(d)
            dnewnew=qgdec.eq16(dnew, LM)
            orgpxl = qgdec.eq9(dnewnew, ka)
        return orgpxl, lsb
   

    

def main():
    image_path = "C:/Users/revel/Documents/GitHub/QGDEC_Steganography/7.1.04.tiff"
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # If the image path is not valid, the img will be None
    if img is None:
        print(f"Image at path {image_path} could not be found.")
        return
    
    mode = input("Choose mode (embed or extract): ")
    if mode == "embed":
        bits = input("Enter bits to embed: ")
        qg = qgdec(img, bits)  # Create an instance of qgdec
        new_block = qg.createblock()  # Pass a block to embedding
    elif mode == "extract":
        LM = input("What's the LM: ")
        qg = qgdec(img, "")  # Create an instance with empty bits for extraction
        block = qg.createblock()  # Get a block
        new_block, bits = qg.extracting(block, LM)
        print(bits)
    else:
        print("Invalid mode. Please choose either embed or extract.")
        return
    
    if 'new_block' in locals():
        # Ensure new_block is a NumPy array and in 8-bit format
        new_block = np.clip(new_block, 0, 255)  # Clip values to [0, 255]
        new_block = np.uint8(new_block)  # Convert to 8-bit format
        
        # Ensure new_block has the correct shape
        height, width = img.shape
        num_elements = height * width
        
        if new_block.size != num_elements:
            print(f"Error: new_block has {new_block.size} elements, expected {num_elements}.")
            return
        
        # Reshape new_block to the same shape as the original image
        new_block = new_block.reshape((height, width))
        
        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(new_block)
        
        # Get the base name of the original image (e.g., '85.jpg')
        base_name = os.path.basename(image_path)
        # Split the base name into name and extension (e.g., '85', '.jpg')
        name, extension = os.path.splitext(base_name)
        # Create a new name by appending '_new' to the name
        new_name = f"{name}_new.png"
        new_image.save(new_name)
        print(f"New image saved as {new_name}")


if __name__=="__main__":
    main()
