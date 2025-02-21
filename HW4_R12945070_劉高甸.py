import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def ADSP_HW4_Jack_R12945070(filename1, filename2):
    X = cv.imread(f"{filename1}.png", cv.IMREAD_GRAYSCALE)
    Y = cv.imread(f"{filename2}.png", cv.IMREAD_GRAYSCALE)
    X_gray_mean = np.mean(X)
    Y_gray_mean = np.mean(Y)
    X_gray_variance = np.mean((X - X_gray_mean) ** 2)
    Y_gray_variance = np.mean((Y - Y_gray_mean) ** 2)
    XY_gray_covariance = np.mean((X - X_gray_mean) * (Y - Y_gray_mean))
    print(f"圖{filename1}的灰階平均： {X_gray_mean:.2f}\n圖{filename2}的灰階平均： {Y_gray_mean:.2f}")
    print(f"圖{filename1}的灰階變異數： {X_gray_variance:.2f}\n圖{filename2}的灰階變異數： {Y_gray_variance:.2f}")
    print(f"圖{filename1} 和 圖{filename2} 的相關係數： {XY_gray_covariance:.2f}")

    L = 255
    c1 = 1 / L ** 0.5
    c2 = 1 / L ** 0.5
    SSIM = (((2 * X_gray_mean * Y_gray_mean + (c1 * L) ** 2) * (2 * XY_gray_covariance + (c2 * L) ** 2)) /
            ((X_gray_mean ** 2 + Y_gray_mean ** 2 + (c1 * L) ** 2) * (X_gray_variance + Y_gray_variance + (c2 * L) ** 2)))
    print(f"圖{filename1} 和 圖{filename2} 的SIMM： is {SSIM:.6f}\n")

    # 畫一張大圖
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Fig{filename1} and Fig{filename2} compare\n\nSSIM (similarity 0~1) : {SSIM:.2f}")

    # 加入第一個子圖
    plt.subplot(1, 2, 1)  # 這個子圖要放在哪裡，plt.subplot(總共的row, 總共的column, 位子)
    plt.axis('off')  # xy軸不要出現
    plt.imshow(X, cmap="gray")  # 加入圖片A，gray是顯示灰階圖，沒這行不會有圖片

    # 加入第二個子圖
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(Y, cmap="gray")

    plt.savefig(f'HW4_{filename1} and {filename2}')
    plt.show()

ADSP_HW4_Jack_R12945070(1, 2)
ADSP_HW4_Jack_R12945070(1, 3)