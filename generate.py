import numpy as np
import pandas as pd
from sys import exit
from matplotlib import image

data = int(123)
usr_input = data

raster = np.zeros([15,15], dtype=np.bool_)

#%% Alingnment

alignment = np.ones([7,7], dtype=np.bool_)
alignment[1:-1, 1:-1] = np.zeros([5,5], dtype=np.bool_)
alignment[2:-2, 2:-2] = np.ones([3,3], dtype=np.bool_)
raster[2:9,2:9] = alignment

#%% Timing

for i in range(8,raster.shape[0]-2,2):
    raster[2,i] = True
    raster[i,2] = True

#%% Numeric Mode

numbers = []

while np.floor(data%10) != 0:
    numbers.append(data%10)
    data = data//10

numbers.reverse()
if len(numbers) > 5:
    print("too many numbers to encode")
    exit()
    
count = np.unpackbits(np.array(len(numbers), dtype=np.uint8))[-3:]

groups = []
remove_zeros = 0

for i in range(len(numbers)//3):
    groups.append(numbers.pop(0)*100+numbers.pop(0)*10+numbers.pop(0))
if len(numbers) == 2:
    groups.append(numbers.pop(0)*10+numbers.pop(0))
    remove_zeros = 3
elif len(numbers) == 1:
    groups.append(numbers.pop(0))
    remove_zeros = 6
    
bits = np.zeros([len(groups),10], dtype=np.bool_)
stream = np.array(count, dtype=np.bool_)
for i, group in enumerate(groups):
    j = 9
    while j > 0:
        bits[i,j] = group%2
        group = group//2
        j -= 1
    if i != len(groups)-1:
        stream = np.append(stream, bits[i])
    else:
        stream = np.append(stream, bits[i,remove_zeros:])
        
while stream.shape[0] < 20:
    stream = np.append(stream, False)
    
#%% Split into codewords

codewords = [stream[:8], stream[8:16], stream[16:]]

def bin2dec(n):
    n = np.flip(n)
    res = 0
    for i,bit in enumerate(n):
        if bit:
            res += 2**i
    return int(res)

def dec2bin(n):
    res = []
    for i in range(8):
        res.append(n%2)
        n = n//2
    return np.flip(res)
            
msg_coefs = []
for word in codewords:
    msg_coefs.append(bin2dec(word))
    
#%% Get error correction codewords

g = [0, 25, 1]
mc = msg_coefs.copy()

logtable = pd.read_csv("logtable.csv", sep=";")["dec"]

def dec2alpha(n):
    return logtable[logtable == n].index[0]

def alpha2dec(n):
    return logtable[n]

for i in range(3):
    if mc[0] != 0:
        first = dec2alpha(mc[0])
        g_new = g.copy()
        for i,(n,m) in enumerate(zip(g_new,mc)):
            g_new[i] = (n+first)%255
            mc[i] = m^alpha2dec(g_new[i])
    mc.pop(0)
    mc.append(0)

err_codewords = mc[0:2]

#%% Place words in raster

def place2(x,y,data1,data2):
    raster[y,x] = bool(data1)
    raster[y,x-1] = bool(data2)

def place_bru(x,y,data):
    for i in range(len(data)//2):
        place2(x,y-i,data[i*2],data[i*2+1])
        
def place_trl(x,y,data):
    place2(x,y,data[0],data[1])
    place2(x,y+1,data[2],data[3])
    place2(x-2,y+1,data[4],data[5])
    place2(x-2,y,data[6],data[7])

    
for i,word in enumerate(codewords):
    place_bru(12,12-i*4,word)
    
for i,word in enumerate(err_codewords):
    place_trl(10-i*4,11,dec2bin(word))
    
#%% Generate possible masking

mask00 = np.zeros((11,11), dtype=np.bool_)
mask01 = np.zeros((11,11), dtype=np.bool_)
mask10 = np.zeros((11,11), dtype=np.bool_)
mask11 = np.zeros((11,11), dtype=np.bool_)

for i in range(mask00.shape[0]):
    for j in range(mask00.shape[0]):
        if i>0 and j>0:
            if i%2 == 0:
                mask00[i,j] = True
            if ((i//2)+(j//3))%2 == 0:
                mask01[i,j] = True
            if ((i*j)%2 + (i*j)%3)%2 == 0:
                mask10[i,j] = True
            if ((i+j)%2 + (i*j)%3)%2 == 0:
                mask11[i,j] = True

masks = [mask00,mask01,mask10,mask11]
for i,mask in enumerate(masks):
    masks[i][0:9,0:9] = False


#%% Evaluate best and apply to raster

def evaluateScore(data, mask):
    data_right = data[:,-1]
    data_lower = data[-1,:]
    mask_right = mask[:,-1]
    mask_lower = mask[-1,:]
    sum1 = 0
    sum2 = 0
    for i,(dr,dl,mr,ml) in enumerate(zip(data_right,data_lower,mask_right,mask_lower)):
        if i>0:
            if dr^mr:
                sum1 += 1
            if dl^ml:
                sum2 += 1
    if sum1>sum2:
        return sum2*16+sum1
    else:
        return sum1*16+sum2
    
scores = np.zeros(4)
for i,mask in enumerate(masks):
    scores[i] = evaluateScore(raster[2:13,2:13],mask)

best_i = np.argmax(scores)
bestmask = masks[best_i]
raster[2:13,2:13] = np.logical_xor(raster[2:13,2:13], bestmask)

#%% Add format information

msg_bits = np.array([0,0,0,best_i//2%2,best_i%2,0,0,0,0,0,0,0,0,0,0], dtype=np.bool_)
bch_bits = msg_bits.copy()
g_bits = np.array([1,0,1,0,0,1,1,0,1,1,1], dtype=np.bool_)

for i in range(5):
    if bch_bits[0]:
        bch_bits[0:11] = np.logical_xor(bch_bits[0:11],g_bits)
    bch_bits = np.delete(bch_bits, 0)

msg_bits[5:15] = bch_bits
xor_bits = np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,1], dtype=np.bool_)
format_bits = np.logical_xor(msg_bits,xor_bits)

raster[10,3:11] = format_bits[0:8]
raster[3:10,10] = np.flip(format_bits[8:15])

#%% Generate image

pixel = np.kron(raster, np.ones((5,5)))
image.imsave('qrcode.png', pixel, cmap="Greys")