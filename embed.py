import cv2
import numpy as np
import math
from PIL import Image
import os
from collections import deque

class qgdec:

    def __init__(self, im, bits):
        self.img = im
        self.bits = bits
        self.avg_pixel_value = np.average(im)
        self.p = 0

    def createblock(self):
        self.Embed_lvl_queue = []
        new_block = []
        old_block = []
        progress = []
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
                characteristic = (h_avg + h_med)/2
                Embed_lvl = np.zeros_like(self.img, dtype=float)  # Ensure Embed_lvl is a float array
                Embed_lvl[i, j] = self.embedlvl(characteristic)
                # self.Embed_lvl_queue.append(Embed_lvl[i, j])
                if self.avg_pixel_value <= 150:
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
                    if self.p == 0:
                        progress.append(h_avg)
                        progress.append(h_med)
                        progress.append(characteristic)
                        progress.append(kb1)
                        progress.append(d1)
                        progress.append(dnew1)
                        progress.append(dnewnew1)
                        progress.append(new_px1)
                        progress.append(Embed_lvl[0,0])
                        self.p = 1
                        
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
                old_block.append(block) 
        with open('orgpx.txt', 'w') as f:
            # Iterate over the data
            for item in old_block:
                # Write each item on a new line
                f.write("%s\n" % item)
        with open('stgpx.txt', 'w') as f:
            # Iterate over the data
            for item in new_block:
                # Write each item on a new line
                f.write("%s\n" % item)
        print(progress)
        print('avg pxl val',self.avg_pixel_value)
        print('block',self.img[0,0])
        print('newblock',new_block[0])
        return new_block
                
    def trapmf(self, x, abcd):
        """
        Trapezoidal membership function generator.

        Parameters
        ----------
        x : 1d array
            Independent variable.
        abcd : 1d array, length 4
            Four-element vector.  Ensure a <= b <= c <= d.

        Returns
        -------
        y : 1d array
            Trapezoidal membership function.
        """
        assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
        a, b, c, d = np.r_[abcd]
        assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                            a <= b <= c <= d.'
        y = np.ones(len(x))

        idx = np.nonzero(x <= b)[0]
        y[idx] = self.trimf(x[idx], np.r_[a, b, b])

        idx = np.nonzero(x >= c)[0]
        y[idx] = self.trimf(x[idx], np.r_[c, c, d])

        idx = np.nonzero(x < a)[0]
        y[idx] = np.zeros(len(idx))

        idx = np.nonzero(x > d)[0]
        y[idx] = np.zeros(len(idx))

        return y


    def trimf(self, x, abc):
        """
        Triangular membership function generator.

        Parameters
        ----------
        x : 1d array
            Independent variable.
        abc : 1d array, length 3
            Three-element vector controlling shape of triangular function.
            Requires a <= b <= c.

        Returns
        -------
        y : 1d array
            Triangular membership function.
        """
        assert len(abc) == 3, 'abc parameter must have exactly three elements.'
        a, b, c = np.r_[abc]     # Zero-indexing in Python
        assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

        y = np.zeros(len(x))

        # Left side
        if a != b:
            idx = np.nonzero(np.logical_and(a < x, x < b))[0]
            y[idx] = (x[idx] - a) / float(b - a)

        # Right side
        if b != c:
            idx = np.nonzero(np.logical_and(b < x, x < c))[0]
            y[idx] = (c - x[idx]) / float(c - b)

        idx = np.nonzero(x == b)
        y[idx] = 1
        return y

    def centroid(self, x, mfx):
        """
        Defuzzification using centroid (`center of gravity`) method.

        Parameters
        ----------
        x : 1d array, length M
            Independent variable
        mfx : 1d array, length M
            Fuzzy membership function

        Returns
        -------
        u : 1d array, length M
            Defuzzified result

        See also
        --------
        skfuzzy.defuzzify.defuzz, skfuzzy.defuzzify.dcentroid
        """

        '''
        As we suppose linearity between each pair of points of x, we can calculate
        the exact area of the figure (a triangle or a rectangle).
        '''

        sum_moment_area = 0.0
        sum_area = 0.0

        # If the membership function is a singleton fuzzy set:
        if len(x) == 1:
            return (x[0] * mfx[0]
                    / np.fmax(mfx[0], np.finfo(float).eps).astype(float))

        # else return the sum of moment*area/sum of area
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]

            # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
            if not (y1 == y2 == 0.0 or x1 == x2):
                if y1 == y2:  # rectangle
                    moment = 0.5 * (x1 + x2)
                    area = (x2 - x1) * y1
                elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                    moment = 2.0 / 3.0 * (x2 - x1) + x1
                    area = 0.5 * (x2 - x1) * y2
                elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                    moment = 1.0 / 3.0 * (x2 - x1) + x1
                    area = 0.5 * (x2 - x1) * y1
                else:
                    moment = ((2.0 / 3.0 * (x2 - x1) * (y2 + 0.5 * y1))
                            / (y1 + y2) + x1)
                    area = 0.5 * (x2 - x1) * (y1 + y2)

                sum_moment_area += moment * area
                sum_area += area

        return (sum_moment_area
                / np.fmax(sum_area, np.finfo(float).eps).astype(float))
                
    def embedlvl(self, characteristics):
            
        categories = {
            'Very small': {'b': -1.0, 'c': 0.0},
            'Small': {'b': 1.0, 'c': 9.5},
            'Small to medium': {'b': 10.5, 'c': 18.0},
            'Medium to large': {'b': 22.0, 'c': 28.0},
            'Large': {'b': 32.0, 'c': 80.0},
            'Very large': {'b': 100.0, 'c': 128.0},
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
        if self.p == 0:
            print(LV)
        categories = {
        'Very small': {'a': -1.0,'b': 0, 'c': 0.25, 'd': 0.5},
        'Small': {'a': 0.25,'b': 0.5, 'c': 1.5, 'd': 1.75},
        'Small to medium': {'a': 1.5,'b': 1.75, 'c': 3.0, 'd': 3.5},
        'Medium to large': {'a': 3.0,'b': 3.5, 'c': 5.0, 'd': 5.5},
        'Large': {'a': 5.0,'b': 5.5, 'c': 7.0, 'd': 7.5},
        'Very large': {'a': 7.0,'b': 7.75, 'c': 8.0, 'd': 9.0},}
        # Add other categories as needed
        
        if LV:
            selected_points = categories[LV]
        if self.p == 0:
            print(selected_points)
        x = np.arange(0, 8.1, 0.1)
        mfx = self.trapmf(x, [selected_points['a'], selected_points['b'], selected_points['c'], selected_points['d']])
        if self.p ==0:
            print(mfx)
        Embedlvl = np.ceil(self.centroid(x, mfx))
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

        # Clean up the bits by removing any tabs or spaces
        self.bits = self.bits.replace('\t', '').replace(' ', '')

        if Embed_lvl == 0:
            new_bits = 0
        elif len(self.bits) < Embed_lvl:
            if self.bits:  # Check if self.bits is not empty
                new_bits = int(self.bits, 2)
            else:
                new_bits = 0
        elif len(self.bits) >= Embed_lvl and Embed_lvl > 0:
            if self.bits:
                # If there are enough bits, take the first Embed_lvl bits
                new_bits = int(self.bits[:Embed_lvl], 2)
                # Remove them from bits
                self.bits = self.bits[Embed_lvl:]
            else:
                new_bits = 0
        
        if self.bits:
            dnewnew = (2 * dnew) + new_bits 
        else:
            dnewnew = 0

        if self.p == 0:
            print('bla', self.bits)

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
    image_path = "C:/Users/revel/Documents/GitHub/QGDEC_Steganography/Pepper.tiff"
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
