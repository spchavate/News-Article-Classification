

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')



#readig dataset

data=pd.read_csv(r"C:\Users\User\Downloads\BBC News Train.csv\BBC News Train.csv")

print(data.shape) 

print(data.head(10)) 
print(data['Category'].unique())

print(data.isna().sum())

sns.countplot(data['Category'])

### add column of name text_size 

data['text_size']=data['Text'].str.len()


max_text=data["text_size"].max()


print(data[data['text_size']==max_text]) 
 


# distribution plot
sns.distplot(data['text_size'])


def create_wordcloud(words):
    wordcloud=WordCloud(width=500, height=200, max_font_size=110, random_state=(21)).generate(words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis='off'
    plt.show()
    
      
subset=data[data['Category']=='business']
text=subset.Text.values
words=' '.join(text)  
create_wordcloud(words)  
    
    
subset=data[data['Category']=='tech']
text=subset.Text.values
words=' '.join(text)  
create_wordcloud(words)  

subset=data[data['Category']=='sport']
text=subset.Text.values
words=' '.join(text)  
create_wordcloud(words)  

subset=data[data['Category']=='politics']
text=subset.Text.values
words=' '.join(text)  
create_wordcloud(words)  


subset=data[data['Category']=='entertainment']
text=subset.Text.values
words=' '.join(text)  
create_wordcloud(words)   

def process_text(text):
    text=text.lower().replace('\n', ' ').replace('\r', '').strip()
    text=re.sub(' +', ' ', text)
    text=re.sub(r'[^\w\s]', '',text)
    
    stop_words=set(stopwords.words('english'))
    word_tokens=word_tokenize(text)
    filtered_sentence=[w for w in word_tokens if not w in stop_words]

    text=' '.join(filtered_sentence)
    return text

data['Text_parsed']= data['Text'].apply(process_text) 


data.head()





#label Encoding


label_encoder= preprocessing.LabelEncoder()
data['Category_Target']=label_encoder.fit_transform(data['Category']) 

########## optional Step 

data.to_csv('BBC_News_Processed.csv') 

######split the data in training and testing 


X_train, X_test, y_train, y_test= train_test_split(data['Text_parsed'], data['Category_Target'],test_size=0.2, random_state=8)


X_train.shape
X_test.shape
y_train.shape
y_test.shape


tfidf=TfidfVectorizer(max_features=(300))

feature_train= tfidf.fit_transform(X_train).toarray()
labels_train= y_train 

feature_test= tfidf.transform(X_test).toarray()
labels_test=y_test



labels_test.shape
labels_train.shape

feature_train.shape
feature_test.shape



## Building ML models
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(feature_train, labels_train)
rfc_predictions=rfc.predict(feature_test)

from sklearn.metrics import accuracy_score, classification_report


print ('Accuracy', accuracy_score(labels_test, rfc_predictions ))
print(classification_report(labels_test, rfc_predictions))


 
