#  Professional Masters in Big Data Program - Simon Fraser University

#  Assignment 5 (Question 5 - a2_p5.py)

#  Submission Date: 29th October 2018
#  Name: Anurag Bejju
#  Student ID: 301369375
#  Professor Name: Greg Mori


import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torchvision.utils as utils
import matplotlib.pyplot as plt
import torchvision.utils as utils
import os
from PIL import Image
import base64
from io import BytesIO
import torch
import numpy as np

# This function returns the html header for the output file
def html_header():
    return ("""
    <html>
    <head>
    <style>

    body {
    font-family: Arial, Helvetica, sans-serif;
    border-collapse: collapse;
    width: 100%;
    background-image: radial-gradient(#54585A, #a6192e);

    }
    #customers2 {
    color: white;
    }
    #customers {
    font-family: Arial, Helvetica, sans-serif;

    border-collapse: collapse;
    width: 80%;
    align: center;
    font-size:14px
    border: none;
    }

    #customers td, #customers th {
    border: none;
    padding: 8px;
    }

    #customers tr{background-color: #f2f2f2;}

    #customers tr:hover {background-color: #ddd;}

    #customers th {
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: left;
    background-color: #54585A;
    color: white;
    border: none;
    }

    </style>
    </head>
    <body>
    <center>
    <table  id="customers2" width = "80%"  align ="center">
    <tr><td colspan="2"><center><h1>Simon Fraser University</h1></center></td> </tr>
    <tr>
    <td width = "80%">
    <h3>Assignment 2 - CMPT 419/726: Machine Learning, Fall 2018</h3><br>
    Name:<b> Anurag Bejju</b><br>
    Student ID: 301369375<br>
    Date: October 25th, 2018<br>
    Professor: Dr. Greg Mori<br>
    <h4>Question 5: Fine Tuning a Pre-Trained Network:</h4>
    </td><td>
    <img src="data:image/jpeg;base64,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" width="100%" height="100%">
    </td>
    </tr>
    </table>

    <table id="customers"  align="center">
         <tr>
           <th>Test Sample Number</th>
           <th>Image</th>
           <th>Class</th>
           <th>Predicted Class</th>
           <th>P(plane)</th>
           <th>P(car)</th>
           <th>P(bird)</th>
           <th>P(cat)</th>
           <th>P(deer)</th>
           <th>P(dog)</th>
           <th>P(frog)</th>
           <th>P(horse)</th>
           <th>P(ship)</th>
           <th>P(truck)</th>

        </tr>
        """)

# This function converts image tensor to PIL format
def convert_to_pil(inputs):
    images_p = inputs / 2 + 0.5
    return(transforms.ToPILImage()(images_p.squeeze()))

# This function converts image from PIL format to base 64 inorder to embed it into a html with out saving the image locally
def image_to_base64(image_t):
    in_mem_file = BytesIO()
    image_t.save(in_mem_file, format = "PNG")
    # reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    base64_encoded_result_bytes = base64.b64encode(img_bytes)
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    return ("data:image/png;base64,"+base64_encoded_result_str)

# This function returns the probabilities for the input to be classified into any of the 10 classes
def get_probability(outputs):
    sm = torch.nn.Softmax(dim=-1)
    probabilities = sm(outputs)
    return(probabilities.detach().numpy())

# This function gets the indicies for images having a fixed sample size for each class
def get_index_values(trainset_value,samp_size):
    class_values = np.zeros((10,), dtype=int)
    index_class = []
    l = 0
    for i in trainset_value:
        j = i[1]
        if class_values[j] < samp_size and l < (samp_size*10):
            index_class.append(j)
            class_values[j] = class_values[j] +1
            l = l + 1
        elif l >= (samp_size*10):
            break
    print (len(index_class))
    return (index_class)
