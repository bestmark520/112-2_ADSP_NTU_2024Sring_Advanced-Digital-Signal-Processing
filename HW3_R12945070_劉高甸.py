import cv2 as cv
import numpy as np

# 讀取輸入影像
input_img = cv.imread('NTU ID R12945070.jpg')

# 檢查影像是否成功讀取
if input_img is None:
    print("錯誤：無法讀取影像。")
else:
    # 將影像轉換為YCrCb色彩空間
    ycrcb_img = cv.cvtColor(input_img, cv.COLOR_BGR2YCrCb)

    # 獲取各個色彩通道
    y, cr, cb = cv.split(ycrcb_img)

    # 執行4:2:0子採樣
    cr = cr[::2, ::2]
    cb = cb[::2, ::2]

    # 將cr和cb升採樣至與y相同的維度
    cr = cv.resize(cr, (y.shape[1], y.shape[0]), interpolation=cv.INTER_NEAREST)
    cb = cv.resize(cb, (y.shape[1], y.shape[0]), interpolation=cv.INTER_NEAREST)

    # 合併子採樣後的色度與完整亮度
    ycrcb_subsampled = cv.merge([y, cr, cb])

    # 將子採樣影像轉換回RGB
    output_img = cv.cvtColor(ycrcb_subsampled, cv.COLOR_YCrCb2BGR)

    # 儲存輸出影像
    cv.imwrite('output_image.jpg', output_img)
    input_img = cv.resize(input_img, (600, 375))
    output_img = cv.resize(output_img, (600, 375))

    # 顯示輸入影像和輸出影像
    cv.imshow('input', input_img)
    cv.imshow('output', output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()