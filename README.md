# JBolt Search,Classifier,Recommendation System

# Search:

**Jbolt Search Engine** is web application that currently has search engine feature based on TF-IDF algorithm where user is able to enter the contributes they would like there future company to have and the result will show the company with related attributes or environment.

**Example** user provides a query ("Good Health Benefits") will returns the name/logo of the company which provides better health benefits as compares to the other company in the dataset.

**Dataset** has 17 fields where company and pros are the most important attribute for this search engine.
![](django_classify/dataset.png)


## Deployment:
**Web Application Link** https://dmsearch.herokuapp.com/
The application is hosted on free hosting site heroku, the deployment is straight forward deploying the site from the github repository but you might get some errors which I encounter when uploading the website on the heroku server.


1. The packages I include were not available directly to Heroku server therefore I had to create a requirement.txt file on the root directory of the project first using pip which will save all the packages required to successfully run the app on the server.

Use the command below to download pip first if you don't have pip already install

      $curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
      $python get-pip.py
Then go to your project folder and then paste the following command to get your requirement.txt
                                  
       $pip freeze > requirements.txt
                                  
After uploading the web also check if you web dyno is set or not if not you need to manually set it I used the following
                                  
       web gunicorn django_classify.wsgi
       
## Code (Search Engine):

Removing the stop words from the data and converting all multiple white-space characters to single whitespace

                 
      stop_words = stopwords.words('english')
      feature = str(feature)
      feature = re.sub('[^a-zA-Z\s]', '', feature)
      feature = [w for w in feature.split() if w not in set(stop_words)]
      return ' '.join(feature)
 Stemming the words altogether so that similar meaning words are not treated as separate words
 
      english_stemmer = SnowballStemmer('english')
      analyzer = CountVectorizer().build_analyzer()
    
 TF-IDF Calculation for the User's Query
 
       query_matrix = count.transform([query])
       query_tfidf = tfidf_transformer.transform(query_matrix)
       
 Cosine Similarity Scores to retrieve the top similar results'
 
       sim_score = cosine_similarity(query_tfidf, train_tfidf)
       
 ## Web Application
 ![](django_classify/search-page.png)
       
 ## References
https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3

https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8

https://github.com/Heetmadhu/Movie-Recommendation/blob/master/MovieSearch.ipynb

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

https://github.com/mohdkashif93/tf-idf-implementation

# Classification

Classification separates observations into groups based on their characteristics. For exampls, students applying to engineering schools could be separated into potential acceptance, maybe accepted, and unlikely expected based on grades, GRE scores, industrial experience, and outstanding activities.
 
**Algorithm Used**
***Multinomial Naive Bayes Classification:***
Naive Bayes is a machine learning algorithm for classification problems. It is basically based on Bayes probability theorem. this algorithm is one of the simplier algorithm and fast to build models and it makes predictions with Naive Bayes algorithm.

****Bayes’ Theorem is stated as:****

***P(h|d) = (P(d|h) * P(h)) / P(d)***

***f1,f2,f3,....fn = fields in the dataset!***
This application classifies company based on the values from the following field
**Work-life balance **
**Ranking **
**Cultural ratings **
**Growth Opportunities ratings **
**Benefits rating **
**Senior Management **



