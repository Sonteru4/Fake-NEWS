# Fake News 📰

The idea is to develop a machine learning program to identify whether an article might be fake news or not.

**Dataset link:** https://www.kaggle.com/c/fake-news/data

**train.csv:** A full training dataset with the following attributes:
* **id:** unique id for a news article
* **title:** the title of a news article
* **author:** author of the news article
* **text:** the text of the article; could be incomplete
* **label:** a label that marks the article as potentially unreliable
    * **1: unreliable**
    * **0: reliable**

* **test.csv:** A testing training dataset with all the same attributes at train.csv without the label.

## Plots for better understanding 📊

### Counplot of the datapoints 
`This is the countplot for the datapoints belonging to a specific class.`
<p align=center>
   <img src="https://github.com/Ankit152/fake-news/blob/master/img/countplot.jpg" height=648>
</p>
From the above plot it is concluded that the dataset is properly balanced.

### Distribution of Title length of the News
`This the distibution of the Length of the Title of the News.` 
<p align=center>
   <img src="https://github.com/Ankit152/fake-news/blob/master/img/titleLenDis.jpg" height=648>
</p>
From the above plot we can conclude that most of the Titles for the News have a length between 10-20 words.

### Distribution of Text length of the News
`This is the distribution of the length of the Text in the News.`
<p align=center>
   <img arc="https://github.com/Ankit152/fake-news/blob/master/img/textLenDis.jpg" height=648>
</p>
From the above plot we can conclude that the distribution is skew as the maximum number of datapoints lies below 2000 words.
