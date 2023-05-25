
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.measure import shannon_entropy
from skimage.color import rgb2ycbcr


'''
自定义的常用函数
'''

def img_abs(fig_input,b,a,db,da):
    'a:水平方向的像素数'
    'b:竖直方向的像素数'
    'da:水平方向的偏移'
    'db:竖直方向的偏移'

    # 图像的半宽度
    cut_x_half = np.floor(0.5*a);
    cut_y_half = np.floor(0.5*b);

    # 图像中心点的偏移量,负号更容易理解，da db 大于零，图片就是向轴的正方向移动
    yshift = - db;
    xshift = - da;

    # 图像的中心
    Center_x = np.ceil(0.5*fig_input.shape[1]) + xshift;
    Center_y = np.ceil(0.5*fig_input.shape[0]) + yshift;

    # 图像的起点坐标
    x_start = Center_x-cut_x_half;
    y_start = Center_y-cut_y_half;

    # 图像的终点坐标
    x_end = x_start +a-1;
    y_end = y_start +b-1;

    # 重新确定中心并裁剪后的图像
    fig_output = fig_input[int(y_start):int(y_end),int(x_start):int(x_end)];
    return fig_output


def psnr_rmse(input,GT):
    if not input.shape == GT.shape:
        raise ValueError('Input images must have the same dimensions.')

    #将图像格式转为float64
    input_data = np.array(input, dtype=np.float64)
    GT_data = np.array(GT, dtype=np.float64)

    # 直接相减，求差值
    diff = GT_data - input_data

    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')

    # 计算RMSE值
    rmse = math.sqrt(np.mean(diff ** 2.))

    # 精度
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps

    # 计算PSNR
    psnr_rmse = 20*math.log10(255.0/rmse)

    return psnr_rmse


def psnr_amse(input,GT):

    # img1 and img2 have range [0, 255]
    if not input.shape == GT.shape:
        raise ValueError('Input images must have the same dimensions.')

    #将图像格式转为float64
    input_data = input.astype(np.float64)
    GT_data = GT.astype(np.float64)

    # 计算AMSE值
    amse = np.mean((input_data - GT_data) ** 2)

    # 精度
    if amse == 0:
        return float('inf')

    # 计算PSNR
    psnr_amse = 20 * math.log10(255.0 / math.sqrt(amse))

    return psnr_amse

def normalize_rgb(data_input):
    dst = np.zeros(data_input.shape,dtype=np.float32)
    v_min=0;v_max = 255
    # data_output = cv.normalize(data_input,dst=dst,alpha=v_min,beta=v_max,norm_type=cv.NORM_MINMAX)
    data_output = cv.normalize(data_input,dst=dst,alpha=1.0,beta=v_max,norm_type=cv.NORM_INF)*255
    # data_output = cv.normalize(data_input,dst=dst,alpha=1.0,beta=v_max,norm_type=cv.NORM_L1)
    # data_output = cv.normalize(data_input,dst=dst,alpha=1.0,beta=v_max,norm_type=cv.NORM_L2)

    # data_output = cv.normalize(data_input,None,0,1,cv.NORM_MINMAX)

    # data_output = (data_input - np.min(np.min(data_input))) / (np.max(np.max(data_input)) - np.min(np.min(data_input)))

    # data_output =

    # data_output = data_input
    return data_output

def equalize_rgb(data_input):
    B,G,R = cv.split(data_input)
    EB = cv.equalizeHist(B)
    EG = cv.equalizeHist(G)
    ER = cv.equalizeHist(R)
    equal_test = cv.merge((EB, EG, ER))  # merge it back

    return equal_test


def check_psnr_ssim_entropy(A,B,fig):
    print('\nin check_psnr_ssim_entropy :')
    print('\tA.shape : ',A.shape)
    print('\tB.shape : ',B.shape)

    psnr_sklearn = compare_psnr(A, B, data_range=None)

    ssim_sklearn = compare_ssim(A, B, win_size=None, gradient=False, data_range=None,
                                channel_axis=2, gaussian_weights=False,
                                full=False) #重点是设置channel_axis=2,
    # ssim_sklearn = compare_ssim(A, B, win_size=None, gradient=False, data_range=None,
    #                             channel_axis=2, multichannel=True, gaussian_weights=False,
    #                             full=False) #重点是设置channel_axis=2
    # skimage.metrics.structural_similarity(im1, im2, win_size=None, gradient=False, data_range=None,
    #                                       channel_axis=None, multichannel=False, gaussian_weights=False, full=False)

    entropy_sklearn_A = shannon_entropy(A)
    entropy_sklearn_B = shannon_entropy(B)

    print('\tpsnr_sklearn : ',psnr_sklearn)
    print('\tssim_sklearn : ',ssim_sklearn)
    print('\tentropy_sklearn_A : ',entropy_sklearn_A)
    print('\tentropy_sklearn_B : ',entropy_sklearn_B)

    '可视化'
    pixel_inch=3; nrows=2; ncols=5; i=1;
    diff_A_B = A-B
    plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True, figsize=(ncols*pixel_inch, nrows*pixel_inch))
    # 600 x 600 像素（先宽度 后高度）
    # 注意这里的宽度和高度的单位是英寸，1英寸=100像素，所以要除以100
    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('A')
    # plt.imshow(A)
    plt.imshow(cv.cvtColor(A, cv.COLOR_BGR2RGB))


    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('B:ref')
    plt.imshow(B)

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('A-B chanel_0 blue')
    plt.imshow(diff_A_B[:,:,0],cmap='gray')
    plt.colorbar()

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('A-B chanel_1 green')
    plt.imshow(diff_A_B[:,:,1],cmap='gray')
    plt.colorbar()

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('A-B chanel_2 red')
    plt.imshow(diff_A_B[:,:,2],cmap='gray')
    plt.colorbar()

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('Hist A')
    color = ('b','g','r')
    for j,col in enumerate(color):
        hist = cv.calcHist([A],[j],None,[256],[0,256])
        plt.plot(hist,color = col)
        # plt.xlim([0,256])

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('Hist B')
    color = ('b','g','r')
    for j,col in enumerate(color):
        hist = cv.calcHist([B],[j],None,[256],[0,256])
        plt.plot(hist,color = col)

    plt.subplot(nrows,ncols,i);i=i+1;
    plt.title('Hist A-B')
    color = ('b','g','r')
    for j,col in enumerate(color):
        hist = cv.calcHist([A-B],[j],None,[256],[0,256])
        plt.plot(hist,color = col)
        print('\tmean(A-B) chanel_[',j,', ',col,'] = ',np.mean(diff_A_B[:,:,j]))

    return psnr_sklearn, ssim_sklearn, entropy_sklearn_A, entropy_sklearn_A


def compare_sensor_recon(img_GT,img_sensor, img_GT_resize2sensor,img_recon, img_GT_resize2recon):
    print('\nin check_psnr :')
    print('\timg_sensor.shape : ',img_sensor.shape)
    print('\timg_GT_resize2sensor.shape : ',img_GT_resize2sensor.shape)

    psnr_rmse_Gr_s = compare_psnr(img_sensor, img_GT_resize2sensor, data_range=None)
    psnr_rmse_Gr_recon = compare_psnr(img_recon, img_GT_resize2recon, data_range=None)

    ssim_rmse_Gr_s = compare_ssim(img_sensor, img_GT_resize2sensor, win_size=None, gradient=False, data_range=None,
                                channel_axis=2, multichannel=True, gaussian_weights=False,
                                  full=False) #重点是设置channel_axis=2
    ssim_rmse_Gr_recon = compare_ssim(img_sensor, img_GT_resize2sensor, win_size=None, gradient=False, data_range=None,
                                  channel_axis=2, multichannel=True, gaussian_weights=False,
                                  full=False)  # 重点是设置channel_axis=2
    # skimage.metrics.structural_similarity(im1, im2, win_size=None, gradient=False, data_range=None,
    #                                       channel_axis=None, multichannel=False, gaussian_weights=False, full=False)

    shannon_entropy(img_sensor)

    psnr_rmse_improve = (psnr_rmse_Gr_recon - psnr_rmse_Gr_s) / psnr_rmse_Gr_s * 100 # (%)
    ssim_rmse_improve = (ssim_rmse_Gr_recon - ssim_rmse_Gr_s) / ssim_rmse_Gr_s * 100 # (%)

    return psnr_rmse_Gr_recon,psnr_rmse_Gr_s,psnr_rmse_improve





def main():
    # 路径设置
    path_abs = 'D:\WYC_Data\\'

    path_sensor = path_abs + 'QHYCCD_Captures\\BackUp\\'
    fname_sensor = '2023-03-25\\newWB_dark_phi1mm_f30mm_200ms_88Gain\\'
    img_name_sensor = 'norm_ROI_2023-03-24-1653_9-newWB_dark_phi1mm_f30mm_200ms_88Gain_00002.png'
    read_path_sensor = path_sensor + fname_sensor + img_name_sensor

    path_recon = 'My_Optical_design\\Single_lens\\BackUp\\20230325\\results\\fig_rec_20230325\\'
    fname = 'IMG.png'

    read_path_recon = path_abs + path_recon + fname

    read_path_GT = r"C:\Users\WYC\Documents\Zemax\IMAFiles\Demo picture -  640 x 480.bmp"
    print('read_path_recon: ', read_path_recon)
    print('read_path_sensor: ', read_path_sensor)
    print('read_path_GT: ', read_path_GT)

    # 读取数据
    img_recon = cv.imread(read_path_recon, 1)
    img_sensor = cv.imread(read_path_sensor, 1)
    img_GT = cv.imread(read_path_GT, 1)

    # 格式转换，
    img_recon = cv.cvtColor(img_recon, cv.COLOR_BGR2RGB)  # 格式转换，
    img_sensor = cv.cvtColor(img_sensor, cv.COLOR_BGR2RGB)  # 格式转换，
    img_GT = cv.cvtColor(img_GT, cv.COLOR_BGR2RGB)  # 格式转换，



    # 基本参数
    img_recon_a = img_recon.shape[0]
    img_recon_b = img_recon.shape[1]
    img_s_a = img_sensor.shape[0]
    img_s_b = img_sensor.shape[1]
    img_GT_a = img_GT.shape[0]
    img_GT_b = img_GT.shape[1]

    print('img_recon.shape[0]: ',img_recon_a)
    print('img_recon.shape[1]: ',img_recon_b)
    print('img_sensor.shape[0]: ',img_s_a)
    print('img_sensor.shape[1]: ',img_s_b)
    print('img_GT.shape[0]: ',img_GT_a)
    print('img_GT.shape[1]: ',img_GT_b)


    # 调整尺寸
    img_GT_resize2sensor = cv.resize(img_GT, (img_s_b, img_s_a)) # 注意图片的第一个维度是y，第二个维度是x；
    img_GT_resize2recon = cv.resize(img_GT, (img_recon_b, img_recon_a)) # 注意图片的第一个维度是y，第二个维度是x；

    # 归一化========================================================
    # img_GT = equalize_rgb(img_GT)
    # img_GT_resize = equalize_rgb(img_GT_resize)
    # img_sensor = equalize_rgb(img_sensor)
    # img_sensor_resize = equalize_rgb(img_sensor_resize)

    img_GT = normalize_rgb(img_GT)
    img_GT_resize2sensor = normalize_rgb(img_GT_resize2sensor)
    img_sensor = normalize_rgb(img_sensor)
    img_recon = normalize_rgb(img_recon)
    # ==============================================================

    psnr_rmse_Gr_recon,psnr_rmse_Gr_s,psnr_rmse_improve = compare_sensor_recon(img_GT,img_sensor, img_GT_resize2sensor,img_recon, img_GT_resize2recon)

    print('img_recon :', img_recon.shape)
    print('img_sensor :', img_sensor.shape)
    print('img_GT :',img_GT.shape)
    print('img_GT_resize2sensor :', img_GT_resize2sensor.shape)
    print('img_GT_resize2recon :', img_GT_resize2recon.shape)


    print('results : ')
    print('psnr_rmse_Gr_s (dB) =', psnr_rmse_Gr_s)
    print('psnr_rmse_Gr_recon (dB) =', psnr_rmse_Gr_recon)
    print('psnr_rmse_improve (%) =', psnr_rmse_improve)


    plt.figure(1)
    plt.title('img_sensor')
    plt.imshow(img_sensor)
    plt.figure(2)
    plt.title('img_recon')
    plt.imshow(img_recon)
    plt.figure(3)
    plt.title('img_GT')
    plt.imshow(img_GT)
    # plt.figure(4)
    # plt.title('img_GT_resize')
    # plt.imshow(img_GT_resize)


if __name__ == '__main__':

    main()

    plt.show()