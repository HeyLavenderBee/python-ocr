import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import pandas
import time
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
from pathlib import Path
from pytesseract import pytesseract

path_tesseract = r"C:\\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_tesseract
text = pytesseract.image_to_string('images/car-plate1.jpg')
ocr_text_result = ""

def get_metadata(image_path):
    print("--------------------------------")
    print("Starting EXIF data extraction....")
    time.sleep(.5)
    img = Image.open(image_path)
    print("Getting exif data....")
    time.sleep(.5)
    if not img._getexif():
        print("No EXIF data found.")
        return "-"
    for tag, value in img._getexif().items():
        print(f"{TAGS.get(tag)}: {value}")
        if tag == 36867:
            print(value)
    print("Completed.")
    print("--------------------------------")
    time.sleep(.5)

def process_image_ocr(image_path):
    print("Starting image processing for OCR...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #reduzir ruido
    edged = cv2.Canny(bfilter, 30, 200) #detectar bordas
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

    time.sleep(.5)
    #achar contornos da placa do carro
    print("Finding contours...")
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    location

    mask = np.zeros(gray.shape, np.uint8)

    #nem todas as imagens tem contornos detectaveis (às vezes não é nem de placa de carro), então usamos try-except
    try:
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        time.sleep(1)
        print("Cropping the image...")
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        result

        time.sleep(.5)
        print("Annotating the image with detected text...")
        text_result = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text_result, (approx[0][0][0], approx[1][0][1]+60), font, 1, (0,255,0), 2, cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        ocr_text_result = text_result
        time.sleep(.5)
        print("Completed image processing for OCR.")
        return ocr_text_result
    except:
        print("No contour detected. Ending process early.")
        return "-"

def append_data_to_csv(file_path):
    file_name = Path(file_path).name
    file_stem = Path(file_path).stem

    print(file_path)
    data = {'NOME IMAGEM': [file_stem], "DATA": [get_metadata(file_path)], "TEXTO EXTRAIDO": [process_image_ocr(file_path)], "IMAGEM": [file_path]}

    df = pandas.DataFrame(data)
    df.to_csv('output_ocr.csv', mode='a', index=False, header=False)

    time.sleep(.1)
    print("Here is the latest data added to output_ocr.csv:")
    print(df)
    print("********************************")

def process_image(file_path):
    get_metadata(file_path)
    process_image_ocr(file_path)
    append_data_to_csv(file_path)

def check_file_path(file_path):
    if not Path(file_path).is_file():
        time.sleep(.5)
        print("--------------- ERROR -----------------")
        time.sleep(.5)
        print("The file path you entered does not exist. Please check the path and try again.")
        time.sleep(.5)
        print("--------------- ERROR -----------------")
        time.sleep(1.5)
        start_program()
    else:
        print("File path verified.")
        process_image(file_path)

def start_program():
    print("********************************")
    time.sleep(.5)
    print("Welcome to the Image EXIF and OCR Extractor Software!")
    time.sleep(.5)
    print("For extracting EXIF data from an image, please insert an image to the 'images' folder from the following formats: JPG and JPEG")
    time.sleep(.8)
    print("Available images in the 'images' folder:")
    entries = os.listdir('images')
    n = 0
    img_array = []
    for entry in entries:
        print(n, " - ", entry)
        img_array.append(entry)
        n += 1
    file_index = input("Enter the index of the image chosen (0, 1, 2, etc): ")
    file_path = "images/" + img_array[int(file_index)]
    print(file_path)
    check_file_path(file_path)
    time.sleep(1)

start_program()

