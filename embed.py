import cv2
import numpy as np
import math
import docopt

class qgdec:


    @staticmethod
    def eq8(block):
        kb = math.floor(block / 4)
        d = block - kb
        return d

    @staticmethod
    def eq9(block):
        ka = math.ceil(block / 4)
        d = ka - block
        return d

    @staticmethod
    def eq10(block):
        if block <= 2:
            dnew= 0
        else:
            dnew = 0
        return dnew
     
    def eq11(dnew):
        dnewnew = 0
        return dnewnew
    
    @staticmethod
    def eq12(dnewnew):
        kb = math.floor(dnewnew / 4)
        stegpx = dnewnew + kb
        return stegpx
    
    @staticmethod
    def eq13(dnewnew):
        ka = math.ceil(dnewnew / 4)
        stegpx = ka - dnewnew
        return stegpx

    @staticmethod        
    def eq14(block):
        lsb = block[-1]
        return int(lsb)
    
    @staticmethod        
    def eq15(d):
        dnew = math.florr(d/2)
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
            d = qgdec.eq8(block)
        else:
            d = qgdec.eq9(block)
        dnew = qgdec.eq10(d)
        dnewnew = qgdec.eq11(dnew)
        if avg_pixblock <= 150:
            new_block = qgdec.eq12(dnewnew)
        else:
            new_block = qgdec.eq13(dnewnew)
        return new_block

    def extracting(block):
        avg_pixblock = np.mean(block)
        if avg_pixblock <= 150:
            d = qgdec.eq8(block)
        else:
            d = qgdec.eq9(block)
        lsb = qgdec.eq14(block)
        dnew= qgdec.eq15(d)
        dnewnew=qgdec.eq16(dnew)
        if avg_pixblock <= 150:
            orgpxl = qgdec.eq8(dnewnew)
        else:
            orgpxl = qgdec.eq9(dnewnew)
        return 
    
    def fuzmemfunc1(x, a, b, c, d):
        first_expression = max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)
        return max(first_expression)




def main():
    args = docopt.docopt(__doc__, version="0.2")
    in_f = args["--in"]
    out_f = args["--out"]
    image_path = "C:\Users\crp8223\Downloads\LSB-Steganography-master\coverimg"
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
            qgdec.fuzmemfunc1(h_avg)
            qgdec.fuzmemfunc1(h_med)
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

    
