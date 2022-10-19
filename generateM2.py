import numpy as np
import pandas as pd
from sys import exit
from matplotlib import image

data = int(20)
usr_input = data
err_corr_words = 5
pixel_size = 5

max_codewords = 10
if err_corr_words == 5:
    max_num = 10
elif err_corr_words == 6:
    max_num = 8
data_mode = 0
n_bits_count = 4
terminator = [0,0,0,0,0]
size = 13

raster = np.zeros([size,size], dtype=np.bool_)

#%% Functions

def bin2dec(n):
    n = np.flip(n)
    res = 0
    for i,bit in enumerate(n):
        if bit:
            res += 2**i
    return int(res)

def dec2bin8(n):
    res = []
    for i in range(8):
        res.append(n%2)
        n = n//2
    return np.flip(res)

def dec2bin10(n):
    res = []
    for i in range(10):
        res.append(n%2)
        n = n//2
    return np.flip(res)

def place2(x,y,data1,data2):
    raster[y,x] = bool(data1)
    raster[y,x-1] = bool(data2)

def place_bru(x,y,data):
    for i in range(len(data)//2):
        place2(x,y-i,data[i*2],data[i*2+1])
        
def place_trd(x,y,data):
    for i in range(len(data)//2):
        place2(x,y+i,data[i*2],data[i*2+1])
        
def place_trl(x,y,data):
    place2(x,y,data[0],data[1])
    place2(x,y+1,data[2],data[3])
    place2(x-2,y+1,data[4],data[5])
    place2(x-2,y,data[6],data[7])
        
#%% Alingnment

alignment = np.ones([7,7], dtype=np.bool_)
alignment[1:-1, 1:-1] = np.zeros([5,5], dtype=np.bool_)
alignment[2:-2, 2:-2] = np.ones([3,3], dtype=np.bool_)
raster[0:7,0:7] = alignment

#%% Timing

for i in range(8,raster.shape[0],2):
    raster[0,i] = True
    raster[i,0] = True

#%% Numeric Mode

numbers = []

while np.floor(data%10) != 0 or data//10 != 0:
    numbers.append(data%10)
    data = data//10

numbers.reverse()
if len(numbers) > max_num:
    print("too many numbers to encode")
    exit()

mode = np.array([data_mode], dtype=np.bool_)    
count = np.unpackbits(np.array(len(numbers), dtype=np.uint8))[-n_bits_count:]

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
    
stream = np.append(mode, count, axis=0)
for i, group in enumerate(groups):
    bits = dec2bin10(group)
    if i != len(groups)-1:
        stream = np.append(stream, bits)
    else:
        stream = np.append(stream, bits[remove_zeros:])
if stream.shape[0] < (max_codewords-err_corr_words)*8-5:
    stream = np.append(stream, [terminator])        
    
#%% Split into codewords

codewords = []
for i in range(stream.shape[0]//8):
    codewords.append(stream[0+i*8:8+i*8])
if stream.shape[0]%8 != 0:
    codewords.append(np.array(stream[(i+1)*8:]))
while len(codewords[-1])<8:
    codewords[-1] = np.append(codewords[-1],np.array([0]),axis=0)
          
msg_coefs = []
for word in codewords:
    msg_coefs.append(bin2dec(word))
    
toggle = True
while len(codewords) < max_codewords-err_corr_words:
    if toggle:
        padding = np.array([1,1,1,0,1,1,1,0])
        codewords.append(padding)
        msg_coefs.append(bin2dec(padding))
        toggle = False
    else:
        padding = np.array([0,0,0,1,0,0,0,1])
        codewords.append(padding)
        msg_coefs.append(bin2dec(padding))
        toggle = True

#%% Get error correction codewords

if err_corr_words == 5:
    g = [0, 113, 164, 166, 119, 10]
elif err_corr_words == 6:
    g = [0, 166, 0, 134, 5, 176, 15]
mc = msg_coefs.copy()
while len(mc)<len(g):
    mc.append(0)

logtable = pd.read_csv("logtable.csv", sep=";")["dec"]

def dec2alpha(n):
    return logtable[logtable == n].index[0]

def alpha2dec(n):
    return logtable[n]

for i,_ in enumerate(codewords):
    if mc[0] != 0:
        first = dec2alpha(mc[0])
        g_new = g.copy()
        for i,(n,m) in enumerate(zip(g_new,mc)):
            g_new[i] = (n+first)%255
            mc[i] = m^alpha2dec(g_new[i])
    mc.pop(0)
    mc.append(0)

err_codewords = mc[0:err_corr_words]

#%% Place words in raster
    
for i,word in enumerate(codewords):
    if i<3:
        place_bru(12,12-i*4,word)
    else:
        place_trd(10,1+(i-3)*4,word)

if err_corr_words == 6:
    place_trd(10,5,dec2bin8(err_codewords[0]))
    err_codewords = err_codewords[1:]
toggle = True
for i,word in enumerate(err_codewords):
    if toggle:
        place_trd(10-i*2,9,dec2bin8(word))
        toggle = False
    else:
        place_bru(8-(i-1)*2,12,dec2bin8(word))
        toggle = True
    
#%% Generate possible masking

mask00 = np.zeros((size,size), dtype=np.bool_)
mask01 = np.zeros((size,size), dtype=np.bool_)
mask10 = np.zeros((size,size), dtype=np.bool_)
mask11 = np.zeros((size,size), dtype=np.bool_)

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
    scores[i] = evaluateScore(raster,mask)

best_i = np.argmax(scores)
bestmask = masks[best_i]
raster = np.logical_xor(raster, bestmask)

#%% Add format information

msg_bits = np.array([0,0,0,best_i//2%2,best_i%2,0,0,0,0,0,0,0,0,0,0], dtype=np.bool_)
if err_corr_words == 5:
    msg_bits[:3] = np.array([0,0,1])
elif err_corr_words == 6:
    msg_bits[:3] = np.array([0,1,0])
    
bch_bits = msg_bits.copy()
g_bits = np.array([1,0,1,0,0,1,1,0,1,1,1], dtype=np.bool_)

for i in range(5):
    if bch_bits[0]:
        bch_bits[0:11] = np.logical_xor(bch_bits[0:11],g_bits)
    bch_bits = np.delete(bch_bits, 0)

msg_bits[5:15] = bch_bits
xor_bits = np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,1], dtype=np.bool_)
format_bits = np.logical_xor(msg_bits,xor_bits)

raster[8,1:9] = format_bits[0:8]
raster[1:8,8] = np.flip(format_bits[8:15])

#%% Add qiuet zone

qz = np.zeros((17,17))
qz[2:15,2:15] = raster

#%% Generate image

pixel = np.kron(qz, np.ones((pixel_size,pixel_size)))
image.imsave('qrcode.png', pixel, cmap="Greys")
