#!/usr/bin/env python
# coding: utf-8

# ## Twitter Sentiment Analysis

# ## Importing Libraries

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from wordcloud import WordCloud, STOPWORDS
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
import re


# ## Loading Dataset

# In[6]:


df=pd.read_csv("train.csv")


# In[8]:


df.head(10)


# ## Renaming the columns

# In[9]:


df.columns=['sentiment','tweet']


# In[10]:


df.head()


# ## Checking dataset information
# 

# In[12]:


df.info()


# ## Check for missing values
# 

# In[13]:


df.isnull().sum()


# In[28]:


df.sample(20)


# ## Check number of tweets with length greater than 5 and less than 5
# 

# In[17]:


sum(df['tweet'].apply(len)>5), sum(df['tweet'].apply(len)<5)


# ## Check sentiment distribution
# 

# In[19]:


df['sentiment'].value_counts()


# ## Visualize sentiment distribution using a pie chart
# 

# In[20]:


df['sentiment'].value_counts().plot(kind='pie',autopct="%1.0f%%")


# ## Wordcloud

# In[27]:


stopwords=set(STOPWORDS)
wordcloud=WordCloud(background_color="white",stopwords=stopwords,max_words=300,max_font_size=40,scale=5).generate(str(df['tweet']))
plt.imshow(wordcloud)


# In[30]:


df.head(20)


# ## Lowercase all text in the 'tweet' column
# 

# In[33]:


def lowercase_text(tweet):
    return tweet.lower()


# In[35]:


df['tweet']=df['tweet'].apply(lambda x: x.lower())


# In[36]:


df


# ## Remove URLs from tweets

# In[37]:


def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Apply the function to the 'tweet' column
df['tweet'] = df['tweet'].apply(remove_urls)


# ## Remove HTML tags from tweets
# 

# In[38]:


def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)

# Apply the function to the 'tweet' column
df['tweet'] = df['tweet'].apply(remove_html_tags)


# ## Remove special characters from tweets
# 

# In[40]:


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Apply the function to the 'tweet' column
df['tweet'] = df['tweet'].apply(remove_special_characters)


# In[41]:


df


# ## Spliting the dataset into training and testing sets
# 

# In[51]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train,y_test = train_test_split (df['sentiment'],df['tweet'], test_size=0.2,random_state=0)


from sklearn.model_selection import train_test_split

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'], test_size=0.2, random_state=0)

# Checking the shapes of the resulting arrays
X_train.shape, X_test.shape


# In[52]:


X_train.shape, X_test.shape


# ## Initializing a pipeline for the model
# 

# In[53]:


clf=Pipeline([('tfid',TfidfVectorizer()),('rfc',RandomForestClassifier(n_jobs=-1))])


# ## Training the model

# In[ ]:


clf.fit(X_train,y_train)


# ## Evaluating the model
# 

# In[55]:


y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))


# In[56]:


pickle.dump(clf,open("twitter_sentiment_analysis.pkl",'wb'))


# In[16]:


clf.predict(['Why are you feeling dejected? Take the quiz:'])


# In[11]:


# Load the trained model
with open("twitter_sentiment_analysis.pkl", 'rb') as f:
    model = pickle.load(f)

# Function to map predicted values to labels
def predict_sentiment(tweet):
    # Make predictions
    prediction = model.predict([tweet])[0]
    
    # Map predicted values to labels
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# In[17]:


tweet = "Why are you feeling dejected? Take the quiz:"
sentiment = predict_sentiment(tweet)
print("Predicted sentiment:", sentiment)


# In[ ]:




