from django.urls import path, include
from api import views

urlpatterns = [
    path('', views.MainPage.as_view()),
    path('predict', views.predict_diabetictype),
]
