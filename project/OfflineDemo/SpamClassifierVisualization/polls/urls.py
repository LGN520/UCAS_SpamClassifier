# polls/urls.py
from django.conf.urls import url

from . import views

urlpatterns = [
#    url(r'^classify/$', views.classify, name='classify'),
    url(r'^$', views.index, name='index'),
    
]


