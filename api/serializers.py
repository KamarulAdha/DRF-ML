from rest_framework import serializers

import pickle


class MainPageSerializer(serializers.Serializer):

    def validate_data(self):
        age = self.validated_data['age']
        bs_fast = self.validated_data['bs_fast']
        bs_pp = self.validated_data['bs_pp']
        plasma_r = self.validated_data['plasma_r']
        plasma_f = self.validated_data['plasma_f']
        hbA1c = self.validated_data['hbA1c']

        age = float(age)
        bs_fast = float(bs_fast)
        bs_pp = float(bs_pp)
        plasma_r = float(plasma_r)
        plasma_f = float(plasma_f)
        hbA1c = float(hbA1c)

        result = [age,bs_fast,bs_pp,plasma_r,plasma_f,hbA1c]
        """Passing data to model & loading the model from disks"""
        model_path = './ml_model/model.pkl'
        classifier = pickle.load(open(model_path, 'rb'))
        prediction = classifier.predict([result])[0]
        conf_score = np.max(classifier.predict_proba([result]))*100
