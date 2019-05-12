from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
from django.template import Context
from utils.app import *


def home(request):
    return render(request, 'blog/home.html')


def search(request):

    if request.method == 'POST':
        query = request.POST['query']
        company, scores, query_tfidf = index(query)

        return render(request, 'blog/search.html', {'data': zip(company, scores)})
    else:
        form = UserCreationForm()
    return render(request, 'blog/search.html')


def recommend(request):

    if request.method == 'POST':
        query = request.POST['query']
        companies = index3(query)
        context = {
            'posts': companies
        }
        return render(request, 'blog/recommend.html', context)
    else:
        form = UserCreationForm()
    return render(request, 'blog/recommend.html')


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
