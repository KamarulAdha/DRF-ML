from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

import pickle
import numpy as np

from api import serializers




# Create your views here.
class MainPage(APIView):
    # serializer_class = serializers.MainPageSerializer

    def get(self, request, format=None):
        """Returns a list of APIView features"""
        Note = [
            'This is an APIView.',
            'It uses HTTP methods as function.',
            'Gives the most control over your application logic',
            'Is mapped manually to URLs',
        ]
        return Response({'Message': 'Hello! Welcome to the Main Page. Please enter the credentials below.', 'Notes:': Note})


@api_view(["POST"])
def predict_diabetictype(request):

    try:
        age = request.data.get('age',None)
        bs_fast = request.data.get('bs_fast',None)
        bs_pp = request.data.get('bs_pp',None)
        plasma_r = request.data.get('plasma_r',None)
        plasma_f = request.data.get('plasma_f',None)
        hbA1c = request.data.get('hbA1c',None)
        fields = [age, bs_fast, bs_pp, plasma_r, plasma_f, hbA1c]
        if not None in fields:
            """Datapreprocessing Convert the values to float"""
            age = float(age)
            bs_fast = float(bs_fast)
            bs_pp = float(bs_pp)
            plasma_r = float(plasma_r)
            plasma_f = float(plasma_f)
            hbA1c = float(hbA1c)
            result = [age,bs_fast,bs_pp,plasma_r,plasma_f,hbA1c]
            """Passing data to model & loading the model from disks"""
            model_path = 'ml_model/model.pkl'
            classifier = pickle.load(open(model_path, 'rb'))
            prediction = classifier.predict([result])[0]
            conf_score =  np.max(classifier.predict_proba([result]))*100
            predictions = {
                'Error': '0',
                'Message': 'Successfully Rendered',
                'Prediction': prediction,
                'Confidence Score': f'{conf_score}%'
            }
        else:
            predictions = {
                'Error': '1',
                'Message': 'Rendering Failed. Invalid Parameters'
            }
    except Exception as e:
        predictions = {
            'Error': '2',
            'Message': str(e)
        }

    return Response(predictions)
