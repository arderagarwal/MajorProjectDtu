import matplotlib 
matplotlib.use('Agg')
import pandas as pd
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from flask import Flask, request , jsonify , render_template , flash , redirect, url_for, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pickle
import os
import matplotlib.pyplot as plt
import random
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from mapbox import Directions, Static
from mapbox import Geocoder
import map_plotter as mp
from mapboxgl.viz import ImageViz
import flask
from io import BytesIO
import base64
import time
name='static/new_plot.jpg'

df = pd.read_csv('pm2.5.csv')
fd=pd.read_csv('pm_10.csv')
data=df.values
np.random.shuffle(data)
data2=fd.values
np.random.shuffle(data2)
x_data=data[:,:3]
y_data=data[:,3]
x_data2=data2[:,:3]
y_data2=data2[:,3]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size=0.3)
model_rf = RandomForestRegressor(n_estimators=600, oob_score=True, random_state=1000)
model_rf.fit(X_train, y_train) 
pred_train_rf= model_rf.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
#print(r2_score(y_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
#print(r2_score(y_test, pred_test_rf))
y_pred=model_rf.predict(X_test)
model_rf.score(x_data,y_data)
model_rf2 = RandomForestRegressor(n_estimators=600, oob_score=True, random_state=1000)
model_rf2.fit(X_train2, y_train2) 
pred_train_rf2= model_rf2.predict(X_train2)
#print(np.sqrt(mean_squared_error(y_train2,pred_train_rf2)))
#print(r2_score(y_train2, pred_train_rf2))

pred_test_rf2 = model_rf2.predict(X_test2)
#print(np.sqrt(mean_squared_error(y_test2,pred_test_rf2)))
#print(r2_score(y_test2, pred_test_rf2))
y_pred2=model_rf2.predict(X_test2)
y_pred2=model_rf2.predict(X_test2)
img_url = 'https://raw.githubusercontent.com/mapbox/mapboxgl-jupyter/master/examples/mosaic' 
# Configuration 
map_type = 'streets-v11'
xref = 400
yref = 400
scale_config = {'offset_x':20,'offset_y':20,
                'text_offset_x':20,'text_offset_y':25,'real_length':100e3}
geocoder = Geocoder(access_token='pk.eyJ1IjoiYW5vdW1haG5hIiwiYSI6ImNrN3hhNnpidDA5cmwzZm8ybTNqbjNleHYifQ.KHevR8RQ5gR0wMB0j7YU3g')
service = Directions(access_token='pk.eyJ1IjoiYW5vdW1haG5hIiwiYSI6ImNrN3hhNnpidDA5cmwzZm8ybTNqbjNleHYifQ.KHevR8RQ5gR0wMB0j7YU3g')
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

 
class ReusableForm(Form):
    source = TextField('Name:', validators=[validators.required()])
    destination=TextField('Name:',validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])

def Route():
    form = ReusableForm(request.form)
    print (form.errors)
    if request.method == 'POST':
        src=request.form['Source']
        
        dest=request.form['Destination Place']
        mode=request.form['Way of Transportation']
        name='static/new_plot'+str(time.time())+'.jpg'
        response=(geocoder.forward(src))
        r2=(geocoder.forward(dest))
        first = response.geojson()['features'][0]
        second=r2.geojson()['features'][0]
        l1=([(coord) for coord in first['geometry']['coordinates']])
        print(l1)
        l2=([(coord) for coord in second['geometry']['coordinates']])
        #print(l2)

        origin = {'type': 'Feature','properties': {'name': src},'geometry': {'type': 'Point','coordinates': l1}}
        destination = {'type': 'Feature','properties': {'name': dest},'geometry': {'type': 'Point','coordinates': l2}}
        response = service.directions([origin, destination],'mapbox.'+str(mode))
        driving_routes = response.geojson()
        #print(driving_routes)
        #viz.show()
        coordinates=np.array(driving_routes['features'][0]['geometry']['coodinates'])
        #print(coordinates)
        x=coordinates[:,0]
        y=coordinates[:,1]
        hour=dt.datetime.today().hour
        t=np.shape(coordinates)
        #print(t[0])
        arr=np.ones((t[0],1))
        arr=hour*arr
        #print(arr)
        pred=np.append(arr,coordinates,axis = 1)
        pred=np.array(pred)
        pm2=model_rf.predict(pred)
        pm10=model_rf2.predict(pred)
        pollution2 =pm2.sum()/t[0]
        pollution10=pm10.sum()/t[0]
        map_config = mp.get_map_config(xref,yref,x.min(),x.max(),y.min(),y.max())
        
        fig = mp.get_figure('pk.eyJ1IjoiYW5vdW1haG5hIiwiYSI6ImNrN3hhNnpidDA5cmwzZm8ybTNqbjNleHYifQ.KHevR8RQ5gR0wMB0j7YU3g',map_type,map_config)
        xt,yt = mp.transform_data(x,y,map_config)
        fig.gca().plot(xt,yt)
        pollution2=round(pollution2,2)
        pollution10=round(pollution10,2)
        plt.title("Pollution is "+str(pollution2)+" pm 2.5 and "+str(pollution10)+"pm 10 ")
        
        #plt.title("la grafica por: "+columna)
        #plt.plot(df.head(10),'--')
        #os.remove('static/new_plot.jpg')
        '''    if filename.startswith('neww_plot'):  # not to remove other images
                os.remove('static/' + filename)'''
        plt.savefig(name, dpi=100,bbox_inches='tight')
        figfile = BytesIO()
        plt.savefig(figfile, format='jpg')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        result = figdata_png
        #plt.close()
        #img.seek(0)
        #plot_url = base64.b64encode(img.getvalue()).decode()
        #plt.show()

        plt.close()
        #plt.tight_layout()
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
        return render_template('Form2.html',form=form,url=name)
    #retrender_template('output.html', result=figdata_png)
    return render_template('Form2.html', form=form)
@app.route('/Navigation',methods=['GET','POST'])
def my_maps():

  mapbox_access_token = 'pk.eyJ1IjoiYW5vdW1haG5hIiwiYSI6ImNrN3hhNnpidDA5cmwzZm8ybTNqbjNleHYifQ.KHevR8RQ5gR0wMB0j7YU3g'


  return render_template('index2.html',
        mapbox_access_token=mapbox_access_token)
@app.after_request
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
  #k,j=Route()
  #plt.show(j)
  app.run()



