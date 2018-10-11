#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2 as cv
from PIL import Image
import h5py


class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        calculate image's dHash
        :param image: PIL.Image
        :return: dHash(string)
        """
        difference = DHash.__difference(image)
        # Converted to 16 binary system(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value=0, no calculation, optimize program
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # end of every 8 bits
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # < 2 bits, filled with 0. (0xf=>0x0f)
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        calculate 2 images' hamming distance.(based on dHash algorithm)
        :param first: Image or dHash(str)
        :param second: Image or dHash(str)
        :return: hamming distance. (greater value means more different)
        """
        # A. calculate dHash's hamming distance
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

        # B. calculate image's hamming distance
        hamming_distance = 0
        image1_difference = DHash.__difference(first)
        image2_difference = DHash.__difference(second)
        for index, img1_pix in enumerate(image1_difference):
            img2_pix = image2_difference[index]
            if img1_pix != img2_pix:
                hamming_distance += 1
        return hamming_distance

    @staticmethod
    def __difference(image):
        """
        *Private method*
        calculate image's pixel difference.
        :param image: PIL.Image
        :return: difference array(consist of 0/1)
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. Grayscale
        grayscale_image = smaller_image.convert("L")
        # 3. Comparison of adjacent pixels
        pixels = list(grayscale_image.getdata())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference

    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        calculate dHash's hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: hamming distance(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")

    @staticmethod
    def is_similar(image1, image2):
        return hamming_distance(image1, image2) < 5

    @staticmethod
    def has_same_image(image_file, base_image_list):
        img = Image.open(image_file)
        for tmp in base_image_list:
            if DHash.is_similar(img, Image.open(tmp)):
                return True
        return False


if __name__ == '__main__':

    # load image
    image1 = Image.open('/home/chenlei/images/feature_detection/7473-luggage.jpg')
    image2 = Image.open('/home/chenlei/images/feature_detection/7473-luggage.jpg')
    # image2 = Image.open('/home/chenlei/images/feature_detection/7553-luggage.jpeg')
    #
    # image1 = Image.open('/home/chenlei/images/feature_detection/6812-luggage 1.jpg')
    # image2 = Image.open('/home/chenlei/images/feature_detection/6812-luggage 2.jpg')

    # image1 = Image.open('/home/chenlei/images/feature_detection/clinic1.png')
    # image2 = Image.open('/home/chenlei/images/feature_detection/clinic5.jpg')

    # image1 = Image.open('/home/chenlei/images/feature_detection/4114-luggage.jpg')
    # image2 = Image.open('/home/chenlei/images/feature_detection/5395-luggage.jpg')

    # if hamming distance < 5, it means the same picture basically
    hamming_distance = DHash.hamming_distance(image1, image2)

    print(hamming_distance)

    dHash1 = DHash.calculate_hash(image1)
    dHash2 = DHash.calculate_hash(image2)
    hamming_distance = DHash.hamming_distance(dHash1, dHash2)

    print(hamming_distance)
