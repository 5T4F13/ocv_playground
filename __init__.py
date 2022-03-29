from flask import Flask, render_template, request
import cv2
import numpy as np
from collections import OrderedDict
from base64 import b64encode
import os
import requests


app = Flask(__name__)
app.debug = True

modifs_order = OrderedDict()

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/ocv', methods=['GET', 'POST'])
def ocv():
    # Select images from folder /images
    path = './images'
    images = os.listdir(path)
    if '.wh..wh..opq' in images:
        images.remove('.wh..wh..opq')
    if request.form.get('img_select'):
        original = cv2.imread(f"images/{request.form.get('img_select')}")
    else:
        original = cv2.imread(os.path.join(path, images[0]))
    scale = 500 / original.shape[1]

    # Resize original image to fit screen
    original = cv2.resize(original, None, fx=scale, fy=scale)

    # Modified image initializations
    modified_img = original.copy()

    # Get image limits
    if request.form.get('scaling'):
        tx_max = int(original.shape[1] * int(request.form.get('fx')) / 100)
        ty_max = int(original.shape[0] * int(request.form.get('fy')) / 100)
    else:
        tx_max = original.shape[1]
        ty_max = original.shape[0]


    # Encode original as bytes
    frame = cv2.imencode('.png', original)[1].tobytes()
    original_bytes = b64encode(frame).decode("utf-8")


    # Default choices for dropdown menus
    if request.form.get('img_select'):
        current_image = request.form.get('img_select')
    else:
        current_image = images[0]

    interp_dict = {'INTER_LINEAR' : cv2.INTER_LINEAR, 'INTER_NEAREST': cv2.INTER_NEAREST, 'INTER_AREA' : cv2.INTER_AREA, 
    'INTER_CUBIC' : cv2.INTER_CUBIC, 'INTER_LANCZOS4' : cv2.INTER_LANCZOS4}
    if request.form.get('Interpolation'):
        interpolation = interp_dict[request.form['Interpolation']]
        interpolation2display = request.form.get('Interpolation')
    else:
        interpolation = cv2.INTER_LINEAR
        interpolation2display = 'INTER_LINEAR'
    
    thr1_dict = {'THRESH_BINARY' : cv2.THRESH_BINARY, 'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV, 'THRESH_TRUNC' : cv2.THRESH_TRUNC, 
    'THRESH_TOZERO' : cv2.THRESH_TOZERO, 'THRESH_TOZERO_INV' : cv2.THRESH_TOZERO_INV}
    if request.form.get('thr1_type'):
        thr1_type = thr1_dict[request.form['thr1_type']]
        thr1_type2display = request.form.get('thr1_type')
    else:
        thr1_type = cv2.THRESH_BINARY
        thr1_type2display = 'THRESH_BINARY'

    adm_dict = {'ADAPTIVE_THRESH_MEAN_C' : cv2.ADAPTIVE_THRESH_MEAN_C, 'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
    if request.form.get('adm_type'):
        adm_type = adm_dict[request.form.get('adm_type')]
        adm_type2display = request.form.get('adm_type')
    else:
        adm_type = cv2.ADAPTIVE_THRESH_MEAN_C
        adm_type2display = 'ADAPTIVE_THRESH_MEAN_C'


    thr2_dict = {'THRESH_BINARY' : cv2.THRESH_BINARY, 'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV}
    if request.form.get('thr2_type'):
        thr2_type = thr2_dict[request.form['thr2_type']]
        thr2_type2display = request.form.get('thr2_type')
    else:
        thr2_type = cv2.THRESH_BINARY
        thr2_type2display = 'THRESH_BINARY'

    retr_mode_dict = {'RETR_EXTERNAL' : cv2.RETR_EXTERNAL, 'RETR_LIST' : cv2.RETR_LIST, 'RETR_CCOMP' : cv2.RETR_CCOMP, 
                'RETR_TREE' : cv2.RETR_TREE, 'RETR_FLOODFILL' : cv2.RETR_FLOODFILL}
    if request.form.get('retr_mode'):
        retr_mode = retr_mode_dict[request.form['retr_mode']]
        retr_mode2display = request.form.get('retr_mode')
    else:
        retr_mode = cv2.RETR_TREE
        retr_mode2display = 'RETR_TREE'

    aprox_method_dict = {'CHAIN_APPROX_NONE' : cv2.CHAIN_APPROX_NONE, 'CHAIN_APPROX_SIMPLE' : cv2.CHAIN_APPROX_SIMPLE}
    if request.form.get('aprox_method'):
        aprox_method = aprox_method_dict[request.form['aprox_method']]
        aprox_method2display = request.form.get('aprox_method')
    else:
        aprox_method = cv2.CHAIN_APPROX_NONE
        aprox_method2display = 'CHAIN_APPROX_NONE'

    tm_matching_dict = {'TM_CCOEFF' : cv2.TM_CCOEFF, 'TM_CCOEFF_NORMED' : cv2.TM_CCOEFF_NORMED, 'TM_CCORR' : cv2.TM_CCORR,
                'TM_CCORR_NORMED' : cv2.TM_CCORR_NORMED, 'TM_SQDIFF' : cv2.TM_SQDIFF, 'TM_SQDIFF_NORMED' : cv2.TM_SQDIFF_NORMED}
    if request.form.get('tm_matching'):
        tm_matching = tm_matching_dict[request.form['tm_matching']]
        tm_matching2display = request.form.get('tm_matching')
    else:
        tm_matching = cv2.TM_CCOEFF
        tm_matching2display = 'TM_CCOEFF'

    # Initialize values
    p11x = int(request.form.get('p11x')) if request.form.get('p11x') else 0
    p11y = int(request.form.get('p11y')) if request.form.get('p11y') else 0
    p12x = int(request.form.get('p12x')) if request.form.get('p12x') else tx_max
    p12y = int(request.form.get('p12y')) if request.form.get('p12y') else 0
    p13x = int(request.form.get('p13x')) if request.form.get('p13x') else tx_max
    p13y = int(request.form.get('p13y')) if request.form.get('p13y') else ty_max
    p21x = int(request.form.get('p21x')) if request.form.get('p21x') else 0
    p21y = int(request.form.get('p21y')) if request.form.get('p22y') else 0
    p22x = int(request.form.get('p22x')) if request.form.get('p21y') else tx_max
    p22y = int(request.form.get('p22y')) if request.form.get('p23x') else 0
    p23x = int(request.form.get('p23x')) if request.form.get('p22x') else tx_max
    p23y = int(request.form.get('p23y')) if request.form.get('p23y') else ty_max
    p14x = int(request.form.get('p14x')) if request.form.get('p14x') else 0
    p14y = int(request.form.get('p14y')) if request.form.get('p14y') else 0
    p15x = int(request.form.get('p15x')) if request.form.get('p15x') else tx_max
    p15y = int(request.form.get('p15y')) if request.form.get('p15y') else 0
    p16x = int(request.form.get('p16x')) if request.form.get('p16x') else 0
    p16y = int(request.form.get('p16y')) if request.form.get('p16y') else ty_max        
    p17x = int(request.form.get('p17x')) if request.form.get('p17x') else tx_max
    p17y = int(request.form.get('p17y')) if request.form.get('p17y') else ty_max
    p24x = int(request.form.get('p24x')) if request.form.get('p24x') else 0
    p24y = int(request.form.get('p24y')) if request.form.get('p24y') else 0
    p25x = int(request.form.get('p25x')) if request.form.get('p25x') else tx_max
    p25y = int(request.form.get('p25y')) if request.form.get('p25y') else 0
    p26x = int(request.form.get('p26x')) if request.form.get('p26x') else 0
    p26y = int(request.form.get('p26y')) if request.form.get('p26y') else ty_max        
    p27x = int(request.form.get('p27x')) if request.form.get('p27x') else tx_max
    p27y = int(request.form.get('p27y')) if request.form.get('p27y') else ty_max
    HUE_min = 0
    HUE_max = 179
    if request.form.get('HUE_min'):
        HUE_min = int(request.form.get('HUE_min'))
    if request.form.get('HUE_max'):
        HUE_max = int(request.form.get('HUE_max'))
    SAT_min = 0
    SAT_max = 255
    if request.form.get('SAT_min'):
        SAT_min = int(request.form.get('SAT_min'))
    if request.form.get('SAT_max'):
        SAT_max = int(request.form.get('SAT_max'))
    VAL_min = 0
    VAL_max = 255
    if request.form.get('VAL_min'):
        VAL_min = int(request.form.get('VAL_min'))
    if request.form.get('VAL_max'):
        VAL_max = int(request.form.get('VAL_max'))
    bs = 3
    if request.form.get('bs'):
        bs = int(request.form.get('bs'))
    C = 0
    if request.form.get('C'):
        C = int(request.form.get('C'))
    k1 = 5
    if request.form.get('k1'):
        k1 = int(request.form.get('k1'))
    k2 = 3
    if request.form.get('k2'):
        k2 = int(request.form.get('k2'))
    k3 = 3
    if request.form.get('k3'):
        k3 = int(request.form.get('k3'))
    k4 = 3
    if request.form.get('k4'):
        k4 = int(request.form.get('k4'))
    d = 3
    if request.form.get('d'):
        d = int(request.form.get('d'))
    sig = 75
    if request.form.get('sig'):
        sig = int(request.form.get('sig'))
    k5 = 5
    if request.form.get('k5'):
        k5 = int(request.form.get('k5'))
    k6 = 5
    if request.form.get('k6'):
        k6 = int(request.form.get('k6'))
    k7 = 5
    if request.form.get('k7'):
        k7 = int(request.form.get('k7'))
    k8 = 5
    if request.form.get('k8'):
        k8 = int(request.form.get('k8'))
    k9 = 5
    if request.form.get('k9'):
        k9 = int(request.form.get('k9'))
    k10 = 5
    if request.form.get('k10'):
        k10 = int(request.form.get('k10'))
    k11 = 5
    if request.form.get('k11'):
        k11 = int(request.form.get('k11'))
    k12 = 5
    if request.form.get('k12'):
        k12 = int(request.form.get('k12'))
    k13 = 5
    if request.form.get('k13'):
        k13 = int(request.form.get('k13'))
    p31x = int(request.form.get('p31x')) if request.form.get('p31x') else 100
    p31y = int(request.form.get('p31y')) if request.form.get('p31y') else 100
    p32x = int(request.form.get('p32x')) if request.form.get('p32x') else 200
    p32y = int(request.form.get('p32y')) if request.form.get('p32y') else 200
    if p32x <= p31x:
        p32x = p31x + 1
    if p32y <= p31y:
        p32y = p31y + 1
    if p31x >= tx_max:
        p31x = tx_max - 1
        p32x = tx_max
    if p31y >= ty_max:
        p31y = ty_max - 1
        p32y = ty_max
    p33x = int(request.form.get('p33x')) if request.form.get('p33x') else 100
    p33y = int(request.form.get('p33y')) if request.form.get('p33y') else 100
    p34x = int(request.form.get('p34x')) if request.form.get('p34x') else 200
    p34y = int(request.form.get('p34y')) if request.form.get('p34y') else 200
    if p34x <= p33x:
        p34x = p33x + 1
    if p34y <= p33y:
        p34y = p33y + 1
    if p33x >= tx_max:
        p33x = tx_max - 1
        p34x = tx_max
    if p33y >= ty_max:
        p33y = ty_max - 1
        p34y = ty_max
    th1 = 100
    if request.form.get('th1'):
        th1 = int(request.form.get('th1'))
    th2 = 200
    if request.form.get('th2'):
        th2 = int(request.form.get('th2'))
    a_s = 3
    if request.form.get('a_s'):
        a_s = int(request.form.get('a_s'))
    th3 = 50
    if request.form.get('th3'):
        th3 = int(request.form.get('th3'))
    rd = 30
    if request.form.get('rd'):
        rd = int(request.form.get('rd'))
    dp = 1
    if request.form.get('dp'):
        dp = float(request.form.get('dp'))
    md = 1
    if request.form.get('md'):
        md = int(request.form.get('md'))
    pr1 = 50
    if request.form.get('pr1'):
        pr1 = int(request.form.get('pr1'))
    pr2 = 30
    if request.form.get('pr2'):
        pr2 = int(request.form.get('pr2'))
    mR = 0
    if request.form.get('mR'):
        mR = int(request.form.get('mR'))
    MR = 0
    if request.form.get('MR'):
        MR = int(request.form.get('MR'))
    itr = 5
    if request.form.get('itr'):
        itr = int(request.form.get('itr'))
    p35x = int(request.form.get('p35x')) if request.form.get('p35x') else 0
    p35y = int(request.form.get('p35y')) if request.form.get('p35y') else 0
    p36x = int(request.form.get('p36x')) if request.form.get('p36x') else tx_max
    p36y = int(request.form.get('p36y')) if request.form.get('p36y') else ty_max
    if p36x <= p35x:
        p36x = p35x + 1
    if p36y <= p35y:
        p36y = p35y + 1
    if p35x >= tx_max:
        p35x = tx_max - 1
        p36x = tx_max
    if p35y >= ty_max:
        p35y = ty_max - 1
        p36y = ty_max

    # Define modifications as functions
    def Grey(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def HSV(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2HSV)
        lower = np.array([HUE_min, SAT_min, VAL_min])
        upper = np.array([HUE_max, SAT_max, VAL_max])
        mask = cv2.inRange(modified_img, lower, upper)
        modified_img = cv2.bitwise_and(modified_img, modified_img, mask= mask)
        return modified_img

    def RGB(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_HSV2BGR)
        return modified_img

    def Scaling(modified_img):
        fx = int(request.form['fx']) / 100
        fy = int(request.form['fy']) / 100
        modified_img = cv2.resize(modified_img, None, fx=fx, fy=fy, interpolation=interpolation)
        return modified_img

    def Translation(modified_img):
        tx = int(request.form['tx'])
        ty = int(request.form['ty'])
        M = np.float32([[1,0,tx],[0,1,ty]])
        rows, cols, _ = modified_img.shape
        modified_img = cv2.warpAffine(modified_img, M, (cols, rows))
        return modified_img

    def Rotation(modified_img):
        modified_img = modified_img
        theta = int(request.form['theta'])
        rows, cols, _ = modified_img.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta,1)
        modified_img = cv2.warpAffine(modified_img, M, (cols, rows))
        return modified_img

    def Affine_transf(modified_img):
        rows, cols, _ = modified_img.shape
        pts1 = np.float32([[p11x, p11y], [p12x, p12y], [p13x, p13y]])
        pts2 = np.float32([[p21x, p21y], [p22x, p22y], [p23x, p23y]])
        M = cv2.getAffineTransform(pts1, pts2)
        modified_img = cv2.warpAffine(modified_img, M, (cols, rows))
        return modified_img

    def Perspective_transf(modified_img):
        rows, cols, _ = modified_img.shape
        pts1 = np.float32([[p14x, p14y], [p15x, p15y], [p16x, p16y], [p17x, p17y]])
        pts2 = np.float32([[p24x, p24y], [p25x, p25y], [p26x, p26y], [p27x, p27y]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        modified_img = cv2.warpPerspective(modified_img, M, (cols, rows))
        return modified_img

    def Simple_Thresholding(modified_img):
        thr1 = int(request.form['thr1'])
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        ret, modified_img = cv2.threshold(modified_img, thresh=thr1, maxval=255, type=thr1_type)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Adaptive_Thresholding(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        modified_img = cv2.adaptiveThreshold(modified_img, maxValue=255, adaptiveMethod=adm_type, thresholdType=thr2_type, blockSize=bs, C=C)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Otsu(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        ret, modified_img = cv2.threshold(modified_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Conv_2D(modified_img):
        kernel = np.ones((k1, k1), np.float32)/25
        modified_img = cv2.filter2D(modified_img, -1, kernel)
        return modified_img

    def Avg_blur(modified_img):
        modified_img = cv2.blur(modified_img, (k2, k2))
        return modified_img

    def Gaussian_blur(modified_img):
        modified_img = cv2.GaussianBlur(modified_img, (k3, k3), 0)
        return modified_img

    def Median_blur(modified_img):
        modified_img = cv2.medianBlur(modified_img, k4)
        return modified_img

    def Bilateral_filter(modified_img):
        modified_img = cv2.bilateralFilter(modified_img, d, sig, sig)
        return modified_img

    def Erosion(modified_img):
        kernel = np.ones((k5, k5), np.uint8)
        modified_img = cv2.erode(modified_img, kernel, iterations=1)
        return modified_img

    def Dilate(modified_img):
        kernel = np.ones((k6, k6), np.uint8)
        modified_img = cv2.dilate(modified_img, kernel, iterations=1)
        return modified_img

    def Opening(modified_img):
        kernel = np.ones((k7, k7), np.uint8)
        modified_img = cv2.morphologyEx(modified_img, cv2.MORPH_OPEN, kernel)
        return modified_img

    def Closing(modified_img):
        kernel = np.ones((k8, k8), np.uint8)
        modified_img = cv2.morphologyEx(modified_img, cv2.MORPH_CLOSE, kernel)
        return modified_img

    def Morph_gradient(modified_img):
        kernel = np.ones((k9, k9), np.uint8)
        modified_img = cv2.morphologyEx(modified_img, cv2.MORPH_GRADIENT, kernel)
        return modified_img

    def Top_hat(modified_img):
        kernel = np.ones((k10, k10), np.uint8)
        modified_img = cv2.morphologyEx(modified_img, cv2.MORPH_TOPHAT, kernel)
        return modified_img

    def Black_hat(modified_img):
        kernel = np.ones((k11, k11), np.uint8)
        modified_img = cv2.morphologyEx(modified_img, cv2.MORPH_BLACKHAT, kernel)
        return modified_img

    def Sobel_x(modified_img):
        modified_img = cv2.Sobel(modified_img, cv2.CV_64F, 1, 0, ksize=k12)
        modified_img = np.uint8(modified_img)
        return modified_img

    def Sobel_y(modified_img):
        modified_img = cv2.Sobel(modified_img, cv2.CV_64F, 0, 1, ksize=k13)
        modified_img = np.uint8(modified_img)
        return modified_img

    def Laplacian(modified_img):
        modified_img = cv2.Laplacian(modified_img, cv2.CV_64F)
        modified_img = np.uint8(modified_img)
        return modified_img

    def Canny_edge(modified_img):
        modified_img = cv2.Canny(modified_img, th1, th2, apertureSize=a_s)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Gaussian_pyramid_down(modified_img):
        modified_img = cv2.pyrDown(modified_img)
        return modified_img

    def Gaussian_pyramid_up(modified_img):
        modified_img = cv2.pyrUp(modified_img)
        return modified_img

    def Find_draw_contours(modified_img):
        thr1 = int(request.form['thr1'])
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        ret, modified_img = cv2.threshold(modified_img, thresh=thr1, maxval=255, type=thr1_type)
        contours, hierarchy = cv2.findContours(modified_img, mode=retr_mode, method=aprox_method)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        cv2.drawContours(modified_img, contours, -1, (0,255,0), 1)
        return modified_img

    def Histogram_equalization(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        modified_img = cv2.equalizeHist(modified_img)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def CLAHE(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        modified_img = clahe.apply(modified_img)
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))     
        return modified_img

    def Backprojection(modified_img):
        roi = modified_img[p31y:p32y, p31x:p32x]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsvt = cv2.cvtColor(modified_img, cv2.COLOR_BGR2HSV)
        roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cv2.filter2D(dst, -1, disc, dst)
        ret, thresh = cv2.threshold(dst, th3, 255, 0)
        thresh = cv2.merge((thresh, thresh, thresh))
        modified_img = cv2.bitwise_and(modified_img, thresh)
        cv2.rectangle(modified_img, (p31x, p31y), (p32x, p32y), 255, 2)
        return modified_img

    def Fourier_transform_HPF(modified_img):
        dft = np.fft.fft2(modified_img, axes=(0,1))
        dft_shift = np.fft.fftshift(dft)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20
        mask = np.zeros_like(modified_img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), rd, (255,255,255), -1)[0]
        mask = 255 - mask
        dft_shift_masked = np.multiply(dft_shift, mask) / 255
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        modified_img = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        modified_img = np.ascontiguousarray(np.abs(modified_img).clip(0, 255), dtype=np.uint8)
        return modified_img

    def Fourier_transform_LPF(modified_img):
        dft = np.fft.fft2(modified_img, axes=(0,1))
        dft_shift = np.fft.fftshift(dft)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20
        mask = np.zeros_like(modified_img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), rd, (255,255,255), -1)[0]
        dft_shift_masked = np.multiply(dft_shift, mask) / 255
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        modified_img = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        modified_img = np.ascontiguousarray(np.abs(modified_img).clip(0, 255), dtype=np.uint8)
        return modified_img

    def Template_matching(modified_img):
        grey_modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        template  = grey_modified_img[p33y:p34y, p33x:p34x]
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(grey_modified_img, template, tm_matching)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if tm_matching in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        if top_left and bottom_right:
            cv2.rectangle(modified_img, top_left, bottom_right, 255, 2)
        return modified_img

    def Hough_line_transform(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        modified_img = cv2.Canny(modified_img, th1, th2, apertureSize=a_s)
        lines = cv2.HoughLines(modified_img, 1, np.pi/180, 200)
        try:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(modified_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        except Exception as e:
            pass
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Prob_hough_transform(modified_img):
        modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        modified_img = cv2.Canny(modified_img, th1, th2, apertureSize=a_s)
        lines = cv2.HoughLinesP(modified_img, 1, np.pi / 180, 100 , minLineLength=100, maxLineGap=10)
        try:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(modified_img, (x1,y1), (x2,y2), (0,255,0), 2)
        except Exception as e:
            pass
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Hough_circle_transform(modified_img):
        img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, md, param1=pr1, param2=pr2, minRadius=mR, maxRadius=MR)
        try:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(modified_img, (i[0], i[1]), i[2], (0,255,0), 2)
                cv2.circle(modified_img, (i[0], i[1]), 2, (0,0,255), 3)
        except Exception as e:
            pass
        return modified_img

    def Watershed(modified_img):
        gray = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh ,cv2.MORPH_OPEN, kernel, iterations = 2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(modified_img, markers)
        modified_img[markers == -1] = [255, 0, 0]
        if len(modified_img.shape) == 2:
            modified_img = cv2.merge((modified_img, modified_img, modified_img))
        return modified_img

    def Foreground(modified_img):
        mask = np.zeros(modified_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (p35x, p35y, p36x, p36y)
        try:
            cv2.grabCut(modified_img, mask, rect, bgdModel, fgdModel, itr, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            modified_img = modified_img * mask2[:,:, np.newaxis]
        except Exception as e:
            pass
        cv2.rectangle(modified_img, (p35x, p35y), (p36x, p36y), 255, 2)
        return modified_img

    # Define modifications options
    funcs_dict = {'grey' : [request.form.get('grey'), 'Grey(modified_img)'], 
                'HSV' : [request.form.get('HSV'), 'HSV(modified_img)'], 
                'RGB' : [request.form.get('RGB'), 'RGB(modified_img)'], 
                'scaling' : [request.form.get('scaling'), 'Scaling(modified_img)'], 
                'translation' : [request.form.get('translation'), 'Translation(modified_img)'], 
                'rotation' : [request.form.get('rotation'), 'Rotation(modified_img)'],
                'affine_transf' : [request.form.get('affine_transf'), 'Affine_transf(modified_img)'], 
                'perspective_transf' : [request.form.get('perspective_transf'), 'Perspective_transf(modified_img)'],
                'simple_Thresholding' : [request.form.get('simple_Thresholding'), 'Simple_Thresholding(modified_img)'],
                'adaptive_Thresholding' : [request.form.get('adaptive_Thresholding'), 'Adaptive_Thresholding(modified_img)'],
                'otsu' : [request.form.get('otsu'), 'Otsu(modified_img)'],
                'conv_2D' : [request.form.get('conv_2D'), 'Conv_2D(modified_img)'],
                'avg_blur' : [request.form.get('avg_blur'), 'Avg_blur(modified_img)'],
                'gaussian_blur' : [request.form.get('gaussian_blur'), 'Gaussian_blur(modified_img)'],
                'median_blur' : [request.form.get('median_blur'), 'Median_blur(modified_img)'],
                'bilateral_filter' : [request.form.get('bilateral_filter'), 'Bilateral_filter(modified_img)'],
                'erosion' : [request.form.get('erosion'), 'Erosion(modified_img)'],
                'dilation' : [request.form.get('dilation'), 'Dilate(modified_img)'],
                'opening' : [request.form.get('opening'), 'Opening(modified_img)'],
                'closing' : [request.form.get('closing'), 'Closing(modified_img)'],
                'morph_gradient' : [request.form.get('morph_gradient'), 'Morph_gradient(modified_img)'],
                'top_hat' : [request.form.get('top_hat'), 'Top_hat(modified_img)'],
                'black_hat' : [request.form.get('black_hat'), 'Black_hat(modified_img)'],
                'sobel_x' : [request.form.get('sobel_x'), 'Sobel_x(modified_img)'],
                'sobel_y' : [request.form.get('sobel_y'), 'Sobel_y(modified_img)'],
                'laplacian' : [request.form.get('laplacian'), 'Laplacian(modified_img)'],
                'canny_edge' : [request.form.get('canny_edge'), 'Canny_edge(modified_img)'], 
                'g_p_down' : [request.form.get('g_p_down'), 'Gaussian_pyramid_down(modified_img)'],
                'g_p_up' : [request.form.get('g_p_up'), 'Gaussian_pyramid_up(modified_img)'],
                'find_draw_contours' : [request.form.get('find_draw_contours'), 'Find_draw_contours(modified_img)'],
                'histogram_equalization' : [request.form.get('histogram_equalization'), 'Histogram_equalization(modified_img)'],
                'clahe' : [request.form.get('clahe'), 'CLAHE(modified_img)'],
                'backprojection' : [request.form.get('backprojection'), 'Backprojection(modified_img)'],
                'fourier_transform_HPF' : [request.form.get('fourier_transform_HPF'), 'Fourier_transform_HPF(modified_img)'],
                'fourier_transform_LPF' : [request.form.get('fourier_transform_LPF'), 'Fourier_transform_LPF(modified_img)'],
                'template_matching' : [request.form.get('template_matching'), 'Template_matching(modified_img)'],
                'hough_line_transform' : [request.form.get('hough_line_transform'), 'Hough_line_transform(modified_img)'],
                'prob_hough_transform' : [request.form.get('prob_hough_transform'), 'Prob_hough_transform(modified_img)'],
                'hough_circle_transform' : [request.form.get('hough_circle_transform'), 'Hough_circle_transform(modified_img)'],
                'watershed' : [request.form.get('watershed'), 'Watershed(modified_img)'],
                'foreground' : [request.form.get('foreground'), 'Foreground(modified_img)']
                }

    # Write in ordered dict modifications order
    for func in funcs_dict.keys():
        if func in modifs_order.keys():
            if not modifs_order[func] and funcs_dict[func][0]:
                modifs_order[func] = funcs_dict[func][0]
            if modifs_order[func] and not funcs_dict[func][0]:
                modifs_order.pop(func)        
        else:
            if funcs_dict[func][0]:
                modifs_order[func] = funcs_dict[func][0]

    # If HSV not active, remove BGR
    if 'HSV' not in modifs_order.keys():
        if 'RGB' in modifs_order.keys():
            modifs_order.pop('RGB')

    # Select modifications from ordered dict
    for key in modifs_order.keys():
        modified_img = eval(funcs_dict[key][1])


    # Return dict with info
    ocv_info = {'current_image' : current_image, 'img_select' : request.form.get('img_select'),
                'include_original' : request.form.get('include_original'), 
                'grey' : request.form.get('grey'), 
                'HSV' : request.form.get('HSV'), 
                    'HUE_min' : HUE_min, 'HUE_max' : HUE_max, 'SAT_min' : SAT_min, 'SAT_max' : SAT_max, 'VAL_min' : VAL_min, 'VAL_max' : VAL_max, 
                'RGB' : request.form.get('RGB'), 
                'scaling' : request.form.get('scaling'), 'fx' : request.form.get('fx'), 'fy' : request.form.get('fy'),
                    'Interpolation' : interpolation2display, 
                'translation' : request.form.get('translation'),'tx' : request.form.get('tx'), 'ty' : request.form.get('ty'),
                    'tx_max' : tx_max, 'ty_max' : ty_max, 
                'rotation' : request.form.get('rotation'), 'theta' : request.form.get('theta'), 
                'affine_transf' : request.form.get('affine_transf'), 
                    'p11x' : p11x, 'p11y' : p11y, 'p12x' : p12x, 'p12y' : p12y, 'p13x' : p13x, 'p13y' : p13y,
                    'p21x' : p21x, 'p21y' : p21y, 'p22x' : p22x, 'p22y' : p22y, 'p23x' : p23x, 'p23y' : p23y, 
                'perspective_transf' : request.form.get('perspective_transf'), 
                    'p14x' : p14x, 'p14y' : p14y, 'p15x' : p15x, 'p15y' : p15y, 'p16x' : p16x, 'p16y' : p16y, 'p17x' : p17x, 'p17y' : p17y,
                    'p24x' : p24x, 'p24y' : p24y, 'p25x' : p25x, 'p25y' : p25y, 'p26x' : p26x, 'p26y' : p26y, 'p27x' : p27x, 'p27y' : p27y,
                'images' : images, 'modifs_order' : list(modifs_order.keys()),
                'simple_Thresholding' : request.form.get('simple_Thresholding'), 'thr1' : request.form.get('thr1'), 
                    'thr1_type' : thr1_type2display,
                'adaptive_Thresholding' : request.form.get('adaptive_Thresholding'), 'adm_type' : adm_type2display, 
                    'thr2_type' : thr2_type2display, 'bs' : bs, 'C' : C,
                'otsu' : request.form.get('otsu'),
                'conv_2D' : request.form.get('conv_2D'), 'k1' : k1,
                'avg_blur' : request.form.get('avg_blur'), 'k2' : k2,
                'gaussian_blur' : request.form.get('gaussian_blur'), 'k3' : k3,
                'median_blur' : request.form.get('median_blur'), 'k4' : k4,
                'bilateral_filter' : request.form.get('bilateral_filter'), 'd' : d, 'sig' : sig,
                'erosion' : request.form.get('erosion'), 'k5' : k5,
                'dilation' : request.form.get('dilation'), 'k6' : k6,
                'opening' : request.form.get('opening'), 'k7' : k7,
                'closing' : request.form.get('closing'), 'k8' : k8,
                'morph_gradient' : request.form.get('morph_gradient'), 'k9' : k9,
                'top_hat' : request.form.get('top_hat'), 'k10' : k10,
                'black_hat' : request.form.get('black_hat'), 'k11' : k11,
                'sobel_x' : request.form.get('sobel_x'), 'k12' : k12,
                'sobel_y' : request.form.get('sobel_y'), 'k13' : k13,
                'laplacian' : request.form.get('laplacian'),
                'canny_edge' : request.form.get('canny_edge'), 'th1' : th1, 'th2' : th2, 'a_s' : a_s,
                'g_p_down' : request.form.get('g_p_down'), 
                'g_p_up' : request.form.get('g_p_up'),
                'find_draw_contours' : request.form.get('find_draw_contours'), 
                    'retr_mode' : retr_mode2display, 'aprox_method' : aprox_method2display,
                'histogram_equalization' : request.form.get('histogram_equalization'),
                'clahe' : request.form.get('clahe'),
                'backprojection' : request.form.get('backprojection'), 
                    'p31x' : p31x, 'p31y' : p31y, 'p32x' : p32x, 'p32y' : p32y, 'th3' : th3,
                'fourier_transform_HPF' : request.form.get('fourier_transform_HPF'),
                'fourier_transform_LPF' : request.form.get('fourier_transform_LPF'), 'rd' : rd,
                'template_matching' : request.form.get('template_matching'),
                    'p33x' : p33x, 'p33y' : p33y, 'p34x' : p34x, 'p34y' : p34y, 'tm_matching' : tm_matching2display,
                'hough_line_transform' : request.form.get('hough_line_transform'),
                'prob_hough_transform' : request.form.get('prob_hough_transform'),
                'hough_circle_transform' : request.form.get('hough_circle_transform'), 
                    'dp' : dp, 'md' : md, 'pr1' : pr1, 'pr2' : pr2, 'mR' : mR, 'MR' : MR,
                'watershed' : request.form.get('watershed'),
                'foreground' : request.form.get('foreground'),
                    'itr' : itr, 'p35x' : p35x, 'p35y' : p35y, 'p36x' : p36x, 'p36y' : p36y
                }

    # Encode modified image as byte
    modified_frame = cv2.imencode('.png', modified_img)[1].tobytes()
    modified_bytes = b64encode(modified_frame).decode("utf-8")

    if request.form.get('include_original'):
        return render_template('ocv.html', original_bytes=original_bytes, modified_bytes=modified_bytes, ocv_info=ocv_info)
    else:
        return render_template('ocv.html', modified_bytes=modified_bytes, ocv_info=ocv_info)
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
