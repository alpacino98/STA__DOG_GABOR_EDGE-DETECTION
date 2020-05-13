from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import imagesc as imgsec
import math
import sys
import cv2



question = sys.argv[1]

def alp_kumbasar_21602607_HW2(question):
    if question == '1':
        print("Question 1 is running...")
        
        mat = loadmat('c2p3.mat')
        counts = mat['counts'].flatten()
        stim = mat['stim']
        stim = np.array(stim)
        counts = np.array(counts)


        # # Question 1 A

        print("Question 1 A")
        STEP_MAX_NO = 10
        sta = STAfinder(stim, counts, STEP_MAX_NO)
        pixel_min = np.min(sta)
        pixel_max = np.max(sta)

        for i in range(STEP_MAX_NO):
            plt.imshow(sta[:,:,i], cmap = 'gray', vmin = pixel_min, vmax = pixel_max)
            stuff_in_string = "STA of step {} before the spike.".format(i + 1)
            plt.title(stuff_in_string)
            plt.show()


        # ## Question 1 B

        print("Question 1 B")

        summer_row = np.zeros((16,10))
        for i in range(16):
            for j in range(10):
                summer_row[i,j] = np.sum(sta[:,i,j])


        summer_row_min = np.min(summer_row)
        summer_row_max = np.max(summer_row)
        plt.imshow(summer_row, cmap = 'gray', vmin =summer_row_min, vmax = summer_row_max)
        plt.xlabel("Step time")
        plt.ylabel("Coloumn of the pixel")
        plt.show()


        # ## Question 1 C

        print("Question 1 C")
        size_count = counts.shape[0]
        projection_stim = np.zeros((size_count))
        for i in range(size_count):
            projection_stim[i] = np.sum(sta[:,:,0] * stim[:,:,i])
        projection_stim_max = np.max(projection_stim)
        projection_stim = projection_stim / projection_stim_max

        num_bins = 100
        plt.hist(projection_stim, num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel("Projection of stimulus for 1st spike == > Normalized")
        plt.ylabel("Counts")
        plt.title("Histogram created for normalized stimulus projection")
        plt.show()

        non_zero_indices = np.where(counts != 0)[0]
        non_zero_count = non_zero_indices.shape[0]

        non_zero_stim_projections = np.zeros((non_zero_count))

        for i in range(non_zero_count):
            non_zero_stim_projections[i] = np.sum(sta[:,:,0] * stim[:,:,non_zero_indices[i] - 1])
        non_zero_stim_projections /= np.max(non_zero_stim_projections)

        num_bins = 100
        plt.hist(non_zero_stim_projections, num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel("Projection of stimulus for 1st non zero spikes == > Normalized")
        plt.ylabel("Counts")
        plt.title("Histogram created for normalized stimulus projection of non zero spikes")
        plt.show()

        colors = ['red', 'blue']
        datas = []
        datas.append(projection_stim)
        datas.append(non_zero_stim_projections)
        plt.hist(datas, num_bins, density=False, histtype='bar', color=colors, label=colors)
        plt.legend(prop={'size': 10})
        plt.title('Bar Graph of both normalized stimulus projection with 1st spikes and 1st non-zer spikes')
        plt.ylabel('Counts')
        plt.xlabel('normalized stimulus projection of both found by nonzero and all 1st spikes')
        plt.show()
    
    elif question == '2':
        print("Question 2 is running..")
        # # Question 2

        # ## Question 2 A
        print("Question 2 A")

        SAMPLE = np.zeros((21,21))
        SIG_C_A = 2
        SIG_S_A = 4

        dog_sample = DOG_fieldFinder( 0, 0, SIG_C_A, SIG_S_A, SAMPLE)
        plt.imshow(dog_sample)
        plt.show()

        # ## Question 2 B

        print("Question 2 B")

        img = cv2.imread("hw2_image.bmp", 0)
        img_np = np.array(img)
        plt.imshow(img_np, cmap="gray")
        plt.title("Original Image in black and white...")
        plt.show()

        response_b = convolve(dog_sample, img_np)

        plt.imshow(response_b, cmap ="gray")
        plt.title("Response Image created with DOG kernel")
        plt.show()

        # ## Question 2 C


        print("Question 2 C")
        edge_img = edgeDetection(response_b, 3)

        plt.imshow(edge_img, cmap="gray")
        plt.title("Edge Detection done with DOG kernel")
        plt.show()

        fig = imgsec.plot(edge_img)


        # ## Question 2 D


        print("Question 2 D")

        SHAPE_KERNEL = np.zeros((21,21))
        THETA = math.pi / 2
        SIGMA_L = 3
        SIGMA_W = 3
        LAMDA = 6
        PHI = 0

        gabor_kernel_90 = Gabor_field_finder(SHAPE_KERNEL, THETA, SIGMA_L, SIGMA_W, PHI, LAMDA, 0, 0)

        plt.imshow(gabor_kernel_90)
        plt.title("Gabro Respective Field with Theta = 90 degree")
        plt.show()

        # ## Question 2 E

        print("Question 2 E")

        response_c = convolve(gabor_kernel_90, img_np)

        plt.imshow(response_c, cmap ="gray")
        plt.title("Response Image created with Gabro Respective Field with Theta = 90 degree")
        plt.show()

        # ## Question 2 F

        print("Question 2 F")

        SHAPE_KERNEL = np.zeros((21,21))
        THETA_0 = 0
        SIGMA_L = 3
        SIGMA_W = 3
        LAMDA = 6
        PHI = 0
        gabor_kernel_0 = Gabor_field_finder(SHAPE_KERNEL, THETA_0, SIGMA_L, SIGMA_W, PHI, LAMDA, 0, 0)

        THETA_30 = math.pi / 6
        gabor_kernel_30 = Gabor_field_finder(SHAPE_KERNEL, THETA_30, SIGMA_L, SIGMA_W, PHI, LAMDA, 0, 0)

        THETA_60 = math.pi / 3
        gabor_kernel_60 = Gabor_field_finder(SHAPE_KERNEL, THETA_60, SIGMA_L, SIGMA_W, PHI, LAMDA, 0, 0)

        THETA_90 = math.pi / 2
        gabor_kernel_90 = Gabor_field_finder(SHAPE_KERNEL, THETA_90, SIGMA_L, SIGMA_W, PHI, LAMDA, 0, 0)

       
        plt.imshow(gabor_kernel_0)
        plt.title("Gabor Respective Field with Theta = 0")
        plt.show()
        
        plt.imshow(gabor_kernel_30)
        plt.title("Gabor Respective Field with Theta = 30")
        plt.show()
        
        plt.imshow(gabor_kernel_60)
        plt.title("Gabor Respective Field with Theta = 60")
        plt.show()
        
        plt.imshow(gabor_kernel_90)
        plt.title("Gabor Respective Field with Theta = 90")
        plt.show()


        response_c_0 = convolve(gabor_kernel_0, img_np)
        response_c_30 = convolve(gabor_kernel_30, img_np)
        response_c_60 = convolve(gabor_kernel_60, img_np)


        
        plt.imshow(response_c_0, cmap = "gray")
        plt.title("Response image created by Gabor Respective Field with Theta = 0")
        plt.show()
        
        plt.imshow(response_c_30, cmap = "gray")
        plt.title("Response image created by Gabor Respective Field with Theta = 30")
        plt.show()
        
        plt.imshow(response_c_60, cmap = "gray")
        plt.title("Response image created by Gabor Respective Field with Theta = 60")
        plt.show()
        
        plt.imshow(response_c, cmap = "gray")
        plt.title("Response image created by Gabor Respective Field with Theta = 90")
        plt.show()
    else:
        print("Please Enter a correct number for question. PS: There are only 2 questions.")




def STAfinder(images, counts, step_no):
    av_collector = np.zeros((np.shape(images)[0], np.shape(images)[1], step_no))
    
    for i in range(np.shape(counts)[0]):
        for j in range(step_no):
            if i > j:
                av_collector[:,:,j] += images[:,:,i - j - 1] * counts[i]
    av_collector /= np.sum(counts)
    return av_collector

def DOGmaker( x, y, sig_c, sig_s):
    divider1 = 2 * (sig_c ** 2)
    divider2 = 2 * (sig_s ** 2)

    exp1 = np.exp(-(x**2 + y**2)/divider1)
    exp2 = np.exp(-(x**2 + y**2)/divider2)

    gaus1 = 1/(divider1 * np.pi) * exp1
    gaus2 = 1/(divider2 * np.pi) * exp2

    return gaus1 - gaus2


def DOG_fieldFinder( x, y, sig_c, sig_s, img_array):
    dog_sample = np.zeros((img_array.shape[0], img_array.shape[1]))
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            dog_sample[x + i, y + j] = DOGmaker( x + i - int(img_array.shape[0] / 2), y + j - int(img_array.shape[1] / 2), sig_c, sig_s)
    return dog_sample

def convolve(kernel, img):
    padding_x = int(kernel.shape[0]/2)
    padding_y = int(kernel.shape[1]/2)

    image_pad = np.zeros((2 * padding_x + img.shape[0], 2 * padding_y + img.shape[1]))
    image_pad[padding_x:img.shape[0] + padding_x, padding_y: img.shape[1] + padding_y] = img

    collector = np.zeros((img.shape))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            collector[i,j] = np.sum(image_pad[i:i +  2* padding_x + 1, j:j + 2 * padding_y + 1] * kernel)
    return collector

def edgeDetection(res_img, threshold):

    collector = res_img

    for i in range(collector.shape[0]):
        for j in range(collector.shape[1]):
            if collector[i,j] >= threshold:
                collector[i,j] = 1
            else:
                collector[i,j] = 0
            
    return collector

def Gabor_finder(theta, sigma_l, sigma_w, phi, lamda, x):

    k = np.array([np.sin(theta), np.cos(theta)])
    ort_k = np.array([np.cos(theta), -np.sin(theta)])
    gabor_kernel = np.exp(-(k.dot(x))**2/(2 * sigma_l**2) - (ort_k.dot(x))**2/(2*sigma_w**2))
    gabor_kernel = gabor_kernel * np.cos(2 * np.pi * ort_k.dot(x) / lamda + phi)
    return gabor_kernel

def Gabor_field_finder(kernel_shape, theta, sigma_l, sigma_w, phi, lamda, x, y):

    kernel_collector = np.zeros(kernel_shape.shape)

    for i in range(kernel_shape.shape[0]):
        for j in range(kernel_shape.shape[1]):
            kernel_collector[x + i, y + j] = Gabor_finder(theta, sigma_l, sigma_w, phi, lamda, np.array([x + i - int(kernel_shape.shape[0] / 2), y + j - int(kernel_shape.shape[1] / 2)]))

    return kernel_collector
    

alp_kumbasar_21602607_HW2(question)
