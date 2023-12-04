# Final Project 
# Tristan Ro
# 100250437

import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
sg.theme('LightGreen')


def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

# contrast function
def adjust_contrast_s_curve(image, contrast):
    
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    return adjusted_image

# Saturation Function
def adjust_saturation_hsv(image, saturation_factor):

    # Convert the image to the HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Adjust the saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    
    # Convert back to the RGB color space
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return adjusted_image

def display_image(og_image, file_name):
    
    # Convert numpy array to data that sg.Graph can understand
    og_image_data = np_im_to_data(og_image)

    # get dimensions from resized image
    height, width, _ = og_image.shape

    # Define the layout
    layout = [[sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key='-ORIGINAL-',
        background_color='white',
        change_submits=True,
        drag_submits=True),
        sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(1, 1),
        graph_top_right=(width, height),
        key='-NEW-',
        background_color='white',
        change_submits=True,
        drag_submits=True)],
        [sg.Button('Exit'),sg.Button('Save'), sg.Button('Vectorize')],
        [sg.Text(f'File: {file_name}, Size: {height} x {width} px')],
        [sg.Text('K-Clusters:'), sg.InputText(key='num_clusters')],
        [sg.Text('Contrast:'), sg.Slider((0.1,3.0), default_value=1.0, orientation ='h', size=(50, 10), key='-SLIDERCONTRAST-', enable_events=True, resolution=0.1)],
        [sg.Text('Saturation:'), sg.Slider((0.1, 3.0), default_value=1.5, orientation='h', size=(50, 10), key='-SLIDERSATURATION-', enable_events=True, resolution=0.1)],
        [sg.Text('Pixel Neighborhood diameter:'), sg.Slider((1,50), default_value=25, orientation ='h', size=(50, 10), key='-SLIDERNEIGHBORHOOD-', enable_events=True)],
        [sg.Text('Sigma Color:'), sg.Slider((0,150), default_value=100, orientation ='h', size=(50, 10), key='-SLIDERSIGMACOLOR-', enable_events=True)],
        [sg.Text('SigmaSpace:'), sg.Slider((0, 150), default_value=100, orientation='h', size=(50, 10), key='-SLIDERSIGMASPACE-', enable_events=True)]]
        
        
    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)    
    window['-ORIGINAL-'].draw_image(data=og_image_data, location=(0, height))

    # Event loop
    while True:

        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        # to visualize before vectorizing
        # contrast
        elif event == '-SLIDERCONTRAST-':
            contrast = values['-SLIDERCONTRAST-']
            adjustedImageContrast = adjust_contrast_s_curve(og_image,contrast)
            window['-NEW-'].draw_image(data=np_im_to_data(adjustedImageContrast), location=(0,height))

        #saturation
        elif event == '-SLIDERSATURATION-':
            saturation = values['-SLIDERSATURATION-']
            adjustedImageSaturation= adjust_saturation_hsv(og_image,saturation)
            window['-NEW-'].draw_image(data=np_im_to_data(adjustedImageSaturation), location=(0,height))

        # vectorize
        elif event == 'Vectorize':
            num_clusters= int(values['num_clusters'])
            pixel_neigh_diameter = saturation = int(values['-SLIDERNEIGHBORHOOD-'])
            sigma_color = saturation = int(values['-SLIDERSIGMACOLOR-'])
            sigma_space = saturation = int(values['-SLIDERSIGMASPACE-'])
            contrast = values['-SLIDERCONTRAST-']
            saturation = values['-SLIDERSATURATION-']

            # adjust contrast and saturation
            # hue was not needed
            img = adjust_contrast_s_curve(og_image, contrast)
            img = adjust_saturation_hsv(img, saturation)

            # Defining input data for clustering
            data = np.float32(img).reshape((-1, 3))

            # Defining criteria
                # max 20 iterations
                # or 1.0 acceptable error
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

            # Applying cv2.kmeans function
            k = num_clusters
            _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)

            # Reshape the output data to the size of input image
            result = center[label.flatten()]
            result = result.reshape(img.shape)

            # Smooth the result
            img_blurred = cv2.medianBlur(result, 3)

            # perform BiLateral filtering
            cartoon = cv2.bilateralFilter(img_blurred, pixel_neigh_diameter, sigma_color, sigma_space)

            window['-NEW-'].draw_image(data=np_im_to_data(cartoon), location=(0, height))
    
        # Save button
        elif event =='Save':
            final = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
            new_filename = sg.popup_get_text('Filename', default_text='test.jpeg')
            if new_filename:
                cv2.imwrite(new_filename, final)
                sg.popup(f'Image saved as {new_filename}')

    window.close()

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    file_name = args.file

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)

    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image1.shape}')

    display_image(image1, file_name)
    
if __name__ == '__main__':
    main()
