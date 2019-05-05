from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='blog-home'),
    path('home', views.home, name='blog-home'),
    path('classify', views.classify, name='blog-classify')


]
