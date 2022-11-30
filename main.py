import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Reading text file
text = open('read.txt', encoding='utf-8').read()
# Converting to lower case
lower_case = text.lower()


# Removing Punctuations
# print(lower_case)
# print(string.punctuation)
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
print(cleaned_text)

# Splitting text into words
# tokenized_words= cleaned_text.split()
tokenized_words = word_tokenize(cleaned_text, "english")
# print(tokenized_words)

# Tokenize words
final_words = []

for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# print(final_words)
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)
w = Counter(emotion_list)
# print(w)

# Sentiment Analysis
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:

        print("Negative Sentiment")
    elif pos > neg:

        print("Positive Sentiment")
    else:

        print("Neutral Sentiment")


sentiment_analyse(cleaned_text)

plt.bar(w.keys(), w.values())
plt.savefig('graph.png')
plt.show()
