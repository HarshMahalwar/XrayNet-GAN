from django.urls import path
from .views import *


urlpatterns = [
    path('Pneumonia/', FileViewPneumonia.as_view(), name="PneumoniaImage"),
    path('Normal/', FileViewNormal.as_view(), name="NormalImage"),

]