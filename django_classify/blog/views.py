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
        company, scores, query_tfidf = index(query)

        return render(request, 'blog/home.html', {'data': zip(company, scores)})
    else:
        form = UserCreationForm()
    return render(request, 'blog/home.html')


def classify(request):
    if request.method == 'POST':
        rating = request.POST['rating']
        workload = request.POST['workload']
        culture = request.POST['culture']
        growth = request.POST['growth']
        benefits = request.POST['benefits']
        support = request.POST['support']

        query = rating + ',' + workload + ',' + culture + ',' + growth + ',' + benefits + ',' + support
        company = index2(query)
        return render(request, 'blog/classify.html', {'data': company})
    else:
        form = UserCreationForm()
    return render(request, 'blog/classify.html')
