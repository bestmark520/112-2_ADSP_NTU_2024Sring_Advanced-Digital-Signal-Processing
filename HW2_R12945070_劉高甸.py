import cmath
import numpy as np
import matplotlib.pyplot as plt
from math import *

# 定義常數
aaa = 8
bbb = 2 * aaa + 1
bbb_PTS = 10000

# Hilbert Transform濾波器函數
def hilbert_transform_filter(x): #理想Hilbert，不是1就是-1
    if x == 0:
        return 0
    elif 0 <= x <= 0.5:
        return -1j
    elif 0.5 < x <= 1:
        return 1j

samples = []
for i in np.arange(0, 1, 1 / bbb): #從0到1，每次增加1/bbb，np.a +range
    samples.append(hilbert_transform_filter(i)) #每個值丟進hilbert_transform_filter後，再放進samples數列

# 添加過渡帶
samples[1] = -0.8j
samples[aaa] = -0.6j
samples[aaa + 1] = 0.6j
samples[2 * aaa] = 0.8j

# 計算r_n
r_1 = np.fft.ifft(samples) #NumPy中Inverse Fast Fourier Transform，IFFT的函數，將頻域信號轉換回時間域信號
r_n = np.concatenate((r_1[ceil(bbb / 2):], r_1[:floor(bbb / 2) + 1]), axis=None)
#np.concatenate 合併 #ceil，取上高斯 #floor，取下高斯

Real_Frequence = []
F = np.arange(0.0, 1.0, 1 / bbb_PTS)
for F_i in F:
    s = 0
    for n in range(-8, 8 + 1):
        s += r_n[n + 8] * cmath.exp(-1j * 2 * pi * F_i * n)  #cmath.exp(x)：計算 e 的 x 次冪，可以理解成e^ax微分後的係數a
    Real_Frequence.append(s.imag)

# 繪製頻率響應圖
plt.plot(F, Real_Frequence)
plt.plot(F, [hilbert_transform_filter(i).imag for i in F])#理想Hilbert，不是1就是-1
plt.title("Frequency Response")
plt.legend(['R(F)', 'H_d(F)'])
plt.show()

# 繪製脈衝響應圖
plt.stem(np.array(range(-8, 8 + 1)), r_n) #繪製長條圖
plt.title("Impulse Response r[n]")
plt.legend(['r[n]'])
plt.show()

'''
aaa和bbb控制濾波器長度,分別是通帶和過渡帶的點數。
增加這些值會增加濾波器的長度,影響脈衝響應和頻率響應曲線的平滑程度。

過渡帶的值(samples[1], samples[aaa]等)可以調整來改變濾波器的過渡行為。

bbb_PTS控制繪製頻率響應的樣本點數,增加此值會使頻率響應曲線更加平滑。
'''