import numpy as np

def fftreal(x, y):

    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("長度不同")

    z = x + 1j * y
    #print(z)
    FZ = np.fft.fft(z)
    #print(f"{FZ}\n")

    m = np.arange(len(z))
    N = len(z)

    Fx = (FZ[m] + np.conj(FZ[(N - m) % N])) /2
    Fy = (FZ[m] - np.conj(FZ[(N - m) % N])) /2j
    #print(Fx)
    #print(f"{Fy}\n")

    Fx_formatted = [f"{value.real:.0f} + {value.imag:.0f}j" for value in Fx]
    Fy_formatted = [f"{value.real:.0f} + {value.imag:.0f}j" for value in Fy]
    print(f"Fx is:{Fx_formatted}\nFy is:{Fy_formatted}\n")

    return Fx, Fy

#fftreal(np.array([1, 2, 3, 4]),np.array([5, 6, 7, 8]))
fftreal(np.array([1, 2, 3, 4, 5]),np.array([6, 7, 8, 9, 10]))

def DFT(x): #驗證用的def

    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    dft_result = np.dot(M, x)

    formatted_result = np.array([f"{value.real:.0f} + {value.imag:.0f}j" for value in dft_result])
    print(formatted_result)

    return dft_result

#DFT(np.array([1, 2, 3, 4]))
#DFT(np.array([5, 6, 7, 8]))

DFT(np.array([1, 2, 3, 4, 5]))
DFT(np.array([6, 7, 8, 9, 10]))

def DFT2(x): #驗證用的def
    Fx = np.fft.fft(x)
    #print(Fx)
    return Fx

#DFT2(np.array([1, 2, 3, 4]))
#DFT2(np.array([5, 6, 7, 8]))

DFT2(np.array([1, 2, 3, 4, 5]))
DFT2(np.array([6, 7, 8, 9, 10]))

'''

1.一開始的分析正確：
FZ                = [10+26j, -4, -2-2j,  -4j]
np.conj(FZ)       = [10-26j, -4, -2+2j,  +4j]
np.conj(FZ[::-1]) = [+4j, -2+2j, -4, 10-26j]

Fx                = [10+0j, -2+2j, -2+0j, -2-2j]
Fy                = [26+0j, -2+2j, -2+0j, -2-2j]
FZ                = [10+26j, -4, -2-2j,  -4j]
FZ = Fx +jFy

2.錯誤的原因：
F1[m] = (F3[m] + F*[N-m])/2
F2[m] = (F3[m] - F*[N-m])/2j
這兩個算式一直刻錯

用GPT會寫錯成
F[0] + F*[3]
F[1] + F*[2]
F[2] + F*[1]
F[3] + F*[0]

    Fx = 0.5 * (Z + np.conj(Z[::-1]))
    Fy = -0.5j * (Z - np.conj(Z[::-1]))
這樣是錯的

3.重新分析算式：
在DFT中，當f[n]為real時，F[m] = F*[N-m]

F[0] = F*[4]
F[1] = F*[3]
F[2] = F*[2]
F[3] = F*[1]

F[0] = F*[5]
F[1] = F*[4]
F[2] = F*[3]
F[3] = F*[2]
F[4] = F*[1]

4.正確的數學算式：
Fx = FZ[m] + np.conj(FZ[N-m])
當f[n] = [1, 2, 3, 4]
F[0] + F*[4]
F[1] + F*[3]
F[2] + F*[2]
F[3] + F*[1]

5.最後卡在：F*[4]沒定義，F*[4]其實是F*[0]
解法：Fx = (FZ[m] + np.conj(FZ[(N - m) % N])) /2

===

20240608，17:00-19:00，參考ADSP講義p.450 p.451，以茲紀念

'''