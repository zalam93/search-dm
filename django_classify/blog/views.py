from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
from django.template import Context
# Create your views here.
from utils.app import *


'''posts = [

    {
        'Class': 'Data Mining Spring 2019',
        'Title': 'Text Based Search',
        'Content': 'Web Application on Job Search',
        'Project': 'Dev Phase I'
    }


]'''


def home(request):

    if request.method == 'POST':
        query = request.POST['query']
        result = index(query)
        result = result.split()
        data = [
            {
                'company': result

            }]

        context = {
            'data': data
        }
        return render(request, 'blog/home.html', context)
    else:
        form = UserCreationForm()
    return render(request, 'blog/home.html')


def output(request):
    if request.method == "POST":
        print(request.POST)
        query = request.POST['query']
        result = index(query)
        #context = Context({'company': result})
    return render(request, 'blog/home.html', result)


def about(request):
    return render(request, 'blog/about.html')
