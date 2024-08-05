import cv2
import numpy as np
import math
import docopt

class qgdec:
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
    
    @staticmethod
    def eq11(dnew): # belom
        dnewnew = dnew * 0
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
    def eq14(block):
        lsb = block[-1]
        return int(lsb)
    
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
        

    def embedding(block):
        avg_pixblock = np.mean(block)
        if avg_pixblock <= 150:
            kb = [b - (b % 4) for b in block]
            d = qgdec.eq8(block , kb)
            dnew = qgdec.eq10(d)
            dnewnew = qgdec.eq11(dnew)
            new_block = qgdec.eq12(dnewnew, kb)
        else:
            ka = [b + abs((b % 4) - 3) for b in block]
            d = qgdec.eq9(block, ka)
            dnew = qgdec.eq10(d)
            dnewnew = qgdec.eq11(dnew)
            new_block = qgdec.eq13(dnewnew, ka)
        return new_block

    def extracting(block):
        avg_pixblock = np.mean(block)
        if avg_pixblock <= 150:
            kb = [b - (b % 4) for b in block]
            d = qgdec.eq8(block, kb)
            lsb = qgdec.eq14(block)
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
    
    def fuzmemfunc1(x, a, b, c, d):
        first_expression = max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)
        return max(first_expression)

    def determineLV(outputfuzz):
        categories = {
        'Very small': {'b': 0., 'c': 0.25},
        'Small': {'b': 0.5, 'c': 1.5},
        'Small to moderate': {'b': 1.75, 'c': 3},
        'Moderate to large': {'b': 3.5, 'c': 5},
        'Large': {'b': 5.5, 'c': 7},
        'Very large': {'b': 7.75, 'c': 8},}
        # Add other categories as needed
        closest_category = None
        smallest_difference = float('inf')
        for category, points in categories.items():
            diff_b = abs(outputfuzz - points['b'])
            diff_c = abs(outputfuzz - points['c'])
            smallest_diff = min(diff_b, diff_c)
            # Update the closest category if the current one is closer
            if smallest_diff < smallest_difference:
                smallest_difference = smallest_diff
                closest_category = category
        return closest_category
    
    def COG(LVA, LVM):
        EV = abs(LVA - LVM)
        return EV

def main():
    image_path = "C:\Users\crp8223\Downloads\LSB-Steganography-master\coverimg"
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE, 0)
    # If the image path is not valid, the img will be None
    if img is None:
        print(f"Image at path {image_path} could not be found.")
        return None
    # Iterate over each pixel in the image
    for i in range(0, img.shape[0], 2):
        blockindex = [i] # GANTI JADI PAIR (figure 4) harus perbaikin
                         # intinya di process sama embed / extract itu per block
        for j in range(0, img.shape[1], 2):
            # Append the pixel value to the list
            block = [img[i, j], img[i, j + 1], img[i + 1, j], img[i + 1, j + 1]]
            # fuzzy difference
            h = [img[i, j] - img[i, j + 1], img[i, j + 1] - img[i + 1, j], img[i, j] - img[i + 1, j], 
                 img[i, j] - img[i + 1, j + 1], img[i + 1, j] - img[i + 1, j + 1], img[i, j + 1] - img[i + 1, j + 1]]
            h_avg = np.mean(h)
            h_med = np.median(h)
            out_avg = qgdec.fuzmemfunc1(h_avg)
            out_med = qgdec.fuzmemfunc1(h_med)
            if args['embed']:
                #Handling lossy format
                out_f, out_ext = out_f.split(".")
                if out_ext in lossy_formats:
                    out_f = out_f + ".png"
                    print("Output file changed to ", out_f)
                res = qgdec.embedding(block)
                blockindex.append(res)
                cv2.imwrite(out_f, res)
            elif args["extract"]:
                raw = qgdec.extracting(block)
                with open(out_f, "wb") as f:
                    f.write(raw)
    lossy_formats = ["jpeg", "jpg"]


if __name__=="__main__":
    main()

    
