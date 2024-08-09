import cv2
import numpy as np
import math
from PIL import Image
import os
from collections import deque


class qgdec:

    def __init__(self, old_block, bits):
        self.old_block = old_block
        self.bits = bits
        self.avg_pixel_value = 151
        self.p = 0
        self.LM = [2,2,2,3]
    
    def createblock(self):
        new_block = []
        progress = []
         # Add an extra column with same PV as last column
        block = [self.old_block[0], self.old_block[1], self.old_block[2], self.old_block[3]]
        if self.avg_pixel_value <= 150:
            kb1 = self.old_block[0] - (self.old_block[0] % 4)
            kb2 = self.old_block[1] - (self.old_block[1] % 4)
            kb3 = self.old_block[2] - (self.old_block[2] % 4)
            kb4 = self.old_block[3] - (self.old_block[3] % 4) 
            # di = bitsi
            d1 = self.eq8(self.old_block[0], kb1)
            d2 = self.eq8(self.old_block[1], kb2)
            d3 = self.eq8(self.old_block[2], kb3)
            d4 = self.eq8(self.old_block[3], kb4)
            dnew1 = self.eq15(d1)
            dnew2 = self.eq15(d2)
            dnew3 = self.eq15(d3)
            dnew4 = self.eq15(d4)
            dnewnew1 = self.eq16(dnew1, self.LM.pop(0))
            dnewnew2 = self.eq16(dnew2, self.LM.pop(0))
            dnewnew3 = self.eq16(dnew3, self.LM.pop(0))
            dnewnew4 = self.eq16(dnew4, self.LM.pop(0))
            new_px1 = self.eq12(dnewnew1, kb1)
            new_px2 = self.eq12(dnewnew2, kb2)
            new_px3 = self.eq12(dnewnew3, kb3)
            new_px4 = self.eq12(dnewnew4, kb4)
            if self.p == 0:
                progress.append(kb1)
                progress.append(d1)
                progress.append(dnew1)
                progress.append(dnewnew1)
                progress.append(new_px1)
                self.p = 1
                
        else:
            ka1 = self.old_block[0] + abs((self.old_block[0] % 4) - 3)
            ka2 = self.old_block[1] + abs((self.old_block[1] % 4) - 3)
            ka3 = self.old_block[2] + abs((self.old_block[2] % 4) - 3)
            ka4 = self.old_block[3] + abs((self.old_block[3] % 4) - 3)
            d1 = self.eq9(self.old_block[0], ka1)
            d2 = self.eq9(self.old_block[1], ka2)
            d3 = self.eq9(self.old_block[2], ka3)
            d4 = self.eq9(self.old_block[3], ka4)
            dnew1 = self.eq15(d1)
            dnew2 = self.eq15(d2)
            dnew3 = self.eq15(d3)
            dnew4 = self.eq15(d4)
            dnewnew1 = self.eq16(dnew1, self.LM.pop(0))
            dnewnew2 = self.eq16(dnew2, self.LM.pop(0))
            dnewnew3 = self.eq16(dnew3, self.LM.pop(0))
            dnewnew4 = self.eq16(dnew4, self.LM.pop(0)) 
            new_px1 = self.eq13(dnewnew1, ka1)
            new_px2 = self.eq13(dnewnew2, ka2)
            new_px3 = self.eq13(dnewnew3, ka3)
            new_px4 = self.eq13(dnewnew4, ka4)
            if self.p == 0:
                progress.append(ka4)
                progress.append(d4)
                progress.append(dnew4)
                progress.append(dnewnew4)
                progress.append(new_px4)
                self.p = 1
        block_pxlval = np.array([new_px1, new_px2, new_px3, new_px4])
        new_block.append(block_pxlval)
        print(progress)
        return new_block

    @staticmethod
    def eq8(block, kb):
        d = block - kb 
        return d

    @staticmethod
    def eq9(block, ka):
        d = ka - block
        return d
    
        
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
    def eq16(block, LM):
        if LM == 0:
            return 0
        elif LM == 2:
            return block + 2 ** math.log2(2 * block + 1) + 1
        elif LM == 1 or LM == 3:
            return block + 2 ** math.log2(2 * block + 1)
        else:
            raise ValueError("Invalid LM value. LM should be '0', '1', '2', or '3'.")


def main():
    old_block = [100,98,106,94]
    qg = qgdec(old_block, "") 
    block = qg.createblock()  
    print(block)
    
if __name__=="__main__":
    main()
