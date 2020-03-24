#test api import
from __future__ import print_function
import requests
import json

import os
from flask import Flask, redirect, url_for, request ,render_template, request, Response, jsonify
import Captionizer
import jsonpickle
import numpy as np
import cv2

app = Flask(__name__) 


   
# method from app .py 
# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
   r = request
   # convert string of image data to uint8
   nparr = np.fromstring(r.data, np.uint8)

   # decode image
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

   cap = str(Captionizer.apply_model_to_image_raw_bytes(img))

   # build a response dict to send back to client
   response = {'message': '{}'.format(cap)
                }

   # encode response using jsonpickle
   response_pickled = jsonpickle.encode(response)

   return Response(response=response_pickled, status=200, mimetype="application/json")


# old logic
@app.route('/upload',methods = ['POST', 'GET']) 
def upload():
   label = request.form['label'] 

   if 'photo' in request.files:

      label = request.form['label'] 
      photo = request.files['photo']
      email = request.form['email'] 
      img_formate = request.form['Radios']  # formate of image
      
      print("image_formate",img_formate)

      label = label + img_formate

      #save the image
      photo.save(os.path.join('M:/projects/flask/Static/images', label))

      #main logic to call captionizer/api

      addr = 'http://localhost:5000'
      test_url = addr + '/api/test'

      # prepare headers for http request
      content_type = 'image/' + img_formate[1:]
      headers = {'content-type': content_type}

      img = cv2.imread('Static/images/' + label)
      
      print("path to read image",'test_images/' + label)
      
      # encode image as jpeg
      _, img_encoded = cv2.imencode(img_formate, img)
      
      # send http request with image and receive response
      response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
      
      # decode response
      print("json response caption ",json.loads(response.text))

      # expected output: {u'message': u'image received. size=124x124'}

      #old
      result = {
         'email' : email,
         'label' : label,
         'caption' : json.loads(response.text)['message']
      }
      
      if photo.filename != '':            
         
         return render_template('display.html', result = result)
      
   else:
      return redirect(url_for('index'))

  
@app.route('/') 
def index(): 
   return render_template('/index.html') 
   
if __name__ == '__main__': 
   app.run(debug=True)
   

''' imp
It’s important to know that the outer double-curly braces are not part of the variable, but the print statement. 
If you access variables inside tags don’t put the braces around them
'''