# JBolt Search Engine

**Jbolt Search Engine** is web application that currently has search engine feature based on TF-IDF algorithm where user is able to enter the contributes they would like there future company to have and the result will show the company with related attributes or environment.

**Example** user provides a query ("Good Health Benefits") will returns the name/logo of the company which provides better health benefits as compares to the other company in the dataset.

**Dataset** has 17 fields where company and pros are the most important attribute for this search engine.
![](django_classify/dataset.png)


### Deployment
**Web Application Link** https://dmsearch.herokuapp.com/
The application is hosted on free hosting site heroku, the deployment is straight forward deploying the site from the github repository but you might get some errors which I encounter when uploading the website on the heroku server.


1. The packages I include were not available directly to Heroku server therefore I had to create a requirement.txt file on the root directory of the project first using pip which will save all the packages required to successfully run the app on the server.

  If you don't have pip then use the command below to download pip first:
                      curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py


  
