from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='blog-home'),
    path('home', views.home, name='blog-home'),
    path('search', views.search, name='blog-search'),
    path('classify', views.classify, name='blog-classify'),
    path('recommend', views.recommend, name='blog-recommend')


]
