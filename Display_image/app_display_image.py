import os
import numpy as np
import cv2
import pylab as p
import matplotlib.cm as cm

import uuid




from flask import Flask, request, render_template, send_from_directory

__author__ = 'reethu'

app = Flask(__name__)




APP_ROOT = os.path.dirname(os.path.abspath(__file__))
images = []
imagearray = []
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/pdf")
def pdf():
    filename = images[-1]
    return render_template("pdf.html",image_name = filename)


@app.route('/pdf/<filename>')
def send_images(filename):
    return send_from_directory("DenoisedImages", filename)

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'noisedImages/')
    
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        
        destination = "/".join([target, "uppload.png"])
        
        print ("Save it to:", destination)
        upload.save(destination)
        img = cv2.imread(destination,cv2.IMREAD_GRAYSCALE)
        
        kernel = np.ones((4,4), np.uint8) 
        img_erode  = 255 - cv2.erode(255 - img, kernel,iterations = 1)
        img_sub = cv2.add(img, - img_erode)
        _, img_thresh = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)
        mask = img_thresh == 0
        filename = 'result_'+str(uuid.uuid4())+'.png'
        filename1 = 'result_'+str(uuid.uuid4())+'.pdf'
        img_final = np.where(mask, img_sub, 255)
        p.figimage(img_final,cmap=cm.Greys_r)
        p.savefig('DenoisedImages/' + filename)
        p.savefig('DenoisedImages/' + filename1,papertype='a5',orientation='portrait')
        images.append(filename1)
        
        imagearray.append(img_final)
                
    return render_template("complete_display_image.html", image_name=filename)



@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("DenoisedImages", filename)



def clean_image(input_img):
    kernel = np.ones((4,4), np.uint8) 
    img_erode  = 255 - cv2.erode(255 - input_img, kernel,iterations = 1)
    img_sub = cv2.add(input_img, - img_erode)
    _, img_thresh = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)
    mask = img_thresh == 0                                     
    img_final = np.where(mask, input_img, 255)
    return img_final


def rmse1(true_images, pred_images):
    result = n = 0
    result += np.sum(true_images.ravel()/255.0 - pred_images.ravel()/255.0**2)
    n += len(true_images.ravel())
    return (result / float(n))**0.5


@app.route("/rmse")
def rmse():
    img = cv2.imread(os.path.join("noisedImages","uppload.png"))
    result = rmse1(img,clean_image(img))
    return render_template("rmse.html" , rmse = result)


    
if __name__ == "__main__":
    app.run()

