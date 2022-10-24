# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import os
import pandas as pd
 
# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]
 
def obtain_face_emdedding(FILEPATH, FILENAME, model):
	# Load the image 
	img = Image.open(f"{FILEPATH}/{FILENAME}")
	resize = (160, 160)
	img = img.resize(resize) 
	face_pixels = asarray(img)
	embedding = get_embedding(model, face_pixels)
	gender, age = get_labels(FILENAME) 
	
	return embedding, gender, age
	

def get_labels(FILENAME):
	temp = FILENAME.split('_')
	age = int(temp[0])
	gender = int(temp[1])
	return gender, age


def create_dataframe(FILEPATH):
	overall_dict = {}
	model = load_model('facenet_keras.h5')
	name_list = []
	face_list = []
	gender_list = []
	age_list = []
	
	for images in os.listdir(FILEPATH):
		embedding, gender, age = obtain_face_emdedding(FILEPATH, images, model)
		name_list.append(images)
		face_list.append(embedding)
		gender_list.append(gender)
		age_list.append(age)
		
	overall_dict['filename'] = name_list
	overall_dict['face_embedding'] = face_list
	overall_dict['gender'] = gender_list
	overall_dict['age'] = age_list

	overall_df = pd.DataFrame.from_dict(overall_dict)
	overall_df.to_csv("data/image_dataframe.csv", index=False, header=True, encoding="utf-8-sig")

	

		







