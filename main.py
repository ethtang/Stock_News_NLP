# Data Scraping
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_news(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_table = soup.find(id='news-table')
    parsed_news = []

    for row in news_table.findAll('tr'):
        title = row.a.get_text()
        date_data = row.td.text.split(' ')
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_news.append([ticker, date, time, title])

    return pd.DataFrame(parsed_news, columns=['ticker', 'date', 'time', 'title'])

news_df = scrape_news('AAPL')
print(news_df.head())

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

news_df = analyze_sentiment(news_df)
print(news_df.head())

# Feature Engineering and Model Training
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example feature and label (dummy data)
X = torch.tensor(news_df['sentiment'].values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(np.random.randn(len(X)), dtype=torch.float32).reshape(-1, 1)  # Replace with actual stock prices

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')