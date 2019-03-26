from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
# Create your views here.
from utils.app import *


posts = [

    {
        'Class': 'Data Mining Spring 2019',
        'Title': 'Text Based Search',
        'Content': 'Web Application on Job Search',
        'Project': 'Dev Phase I'
    }


]


def home(request):
    context = {
        'posts': posts
    }
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        message.success(request)
        return redirect('blog-home')
    else:
        form = UserCreationForm()
    return render(request, 'blog/home.html', context)


def output(request):
    if request.method == "POST":
        print(request.POST)
        query = request.POST['query']
        result = index(query)
        print(result)
        '''cc = {
            'result': result
        }'''
        return HttpResponse('<h5>' + result + '</h5>')


def about(request):
    return render(request, 'blog/about.html')
