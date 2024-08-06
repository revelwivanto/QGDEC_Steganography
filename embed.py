import cv2
import numpy as np
import math
import skfuzzy as fuzz
from PIL import Image
import os

class qgdec:

    def __init__(self, im, bits):
        self.img = im
        self.bits = bits
        self.embedlvl = 0

    def createblock(self):
         # if the image is not even, add an extra row or column
        if self.img.shape[0] % 2 == 1:
            self.img = np.vstack([self.img, self.img[-1, :]])  # Add an extra row with same PV as last row
        if self.img.shape[1] % 2 == 1:
            self.img = np.hstack([self.img, self.img[:, -1].reshape(-1, 1)])  # Add an extra column with same PV as last column
        # Iterate over each pixel in the image
        for i in range(0, self.img.shape[0], 2):
            for j in range(0, self.img.shape[1], 2):
                # Append the pixel value to the list
                block = [self.img[i, j], self.img[i, j + 1], self.img[i + 1, j], self.img[i + 1, j + 1]]

                ## put the code below to embedlvl() method
                # fuzzy difference
                h = [self.img[i, j] - self.img[i, j + 1], self.img[i, j + 1] - self.img[i + 1, j], self.img[i, j] - self.img[i + 1, j], 
                    self.img[i, j] - self.img[i + 1, j + 1], self.img[i + 1, j] - self.img[i + 1, j + 1], self.img[i, j + 1] - self.img[i + 1, j + 1]]
                h_avg = np.mean(h)
                h_med = np.median(h)
                Embed_lvl = np.zeros_like(self.img)
                Embed_lvl [i,j]= np.mean(qgdec.embedlvl(h_avg),qgdec.embedlvl(h_med))
                self.Embed_lvl_queue.append(Embed_lvl)
        return block
    
    def embedlvl(characteristics):
        
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
        Embedlvl = fuzz.centroid(x, mfx)
        return Embedlvl
        
    @staticmethod
    def eq8(block, kb):
        d = [b - k for b, k in zip(block, kb)]
        # The rest of your code should be indented to be part of the eq8 function
        LM = [0] * len(d)  # Initialize LM with the correct size
        for i, val in enumerate(d):
            if val == 0:
                LM[i] = 00
            elif val == 2:
                LM[i] = 10
            else:
                LM[i] = 11
        return d, LM

    @staticmethod
    def eq9(block, ka):
        d = [k - b for k, b in zip(ka, block)]
        return d

    @staticmethod
    def eq10(block):
        if block <= 2:
            dnew= 0
        else:
            dnew = 0
        return dnew
        
    def eq11(self, dnew, Embed_lvl):
        Embed_lvl = Embed_lvl.pop()
        # Check if there are enough bits left
        if len(self.bits) < Embed_lvl:
            # If not, take all remaining bits
            new_bits = int(self.bits, 2)
            # And set bits to an empty string
            self.bits = ''
        else:
            # If there are enough bits, take the first Embed_lvl bits
            new_bits = int(self.bits[:Embed_lvl], 2)
            # And remove them from bits
            self.bits = self.bits[Embed_lvl:]
        
        # Calculate dnewnew
        dnewnew = 2 * dnew + new_bits
        
        return dnewnew
    
    @staticmethod
    def eq12(dnewnew, kb):
        stegpx = dnewnew + kb
        return stegpx
    
    @staticmethod
    def eq13(dnewnew, ka):
        stegpx = [k - d for k, d in zip(dnewnew, ka)]
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
        

    def embedding(self, block):
        avg_pixblock = np.mean(block)
        if avg_pixblock <= 150:
            kb = [b - (b % 4) for b in block]
            d = qgdec.eq8(block , kb)
            dnew = qgdec.eq10(d)
            Embed_lvl = qgdec.embed_lvl(block)
            dnewnew = qgdec.eq11(self.bits,dnew,Embed_lvl)
            new_block = qgdec.eq12(dnewnew, kb)
        else:
            ka = [b + abs((b % 4) - 3) for b in block]
            d = qgdec.eq9(block, ka)
            dnew = qgdec.eq10(d)
            dnewnew = qgdec.eq11(dnew)
            new_block = qgdec.eq13(dnewnew, ka)
        return new_block

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
    image_path = "C:/Users/revel/Documents/GitHub/QGDEC_Steganography/85.jpg"
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # If the image path is not valid, the img will be None
    if img is None:
        print(f"Image at path {image_path} could not be found.")
        return None
    mode = input("Choose mode (embed or extract): ")
    if mode == "embed":
        bits = input("Enter bits to embed: ")
        qgdec(img,bits)
        block = qgdec.createblock()
        new_block = qgdec.embedding(block)
    elif mode == "extract":
        LM = input("whats the LM:")
        new_block, bits = qgdec.extracting(block, LM)
        print(bits)
    else:
        print("Invalid mode. Please choose either embed or extract.")
    
    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(new_block)
    
    # Get the base name of the original image (e.g., '85.jpg')
    base_name = os.path.basename(image_path)
    # Split the base name into name and extension (e.g., '85', '.jpg')
    name, extension = os.path.splitext(base_name)
    # Create a new name by appending '_new' to the name
    new_name = f"{name}_new.png"
    new_image.save(new_name)

if __name__=="__main__":
    main()

    
