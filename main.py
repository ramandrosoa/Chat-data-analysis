
import streamlit as st 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import calendar
from os import path
from PIL import Image
import datetime
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.io as pio


plt.style.use('ggplot')

st.set_page_config(page_title="WhatsApp Chat Data Analysis", layout='centered')
st.title('WhatsApp Chat Data Analysis')

def get_file_list(folder_path, exclude_files = ["requirements.txt"]):
    return [f for f in os.listdir(folder_path)
            if f.endswith('.txt') and f not in exclude_files]

#Preprocessing
def preprocess(file):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            with open(file, 'r', encoding=encoding) as f:
                chat = ' '.join(f.read().split('\n'))
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error(f"Unable to decode the file. Tried encodings: {', '.join(encodings)}")
        return None

    user_msg = re.split('\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s', chat)[1:]
    date_time = re.findall('\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s', chat)

    chat_df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})

    def split_user_msg(row):
        part = row.split(':', 1)
        if len(part) == 2:
            return pd.Series({'user': part[0].strip(), 'text': part[1].strip()})
        else:
            return pd.Series({'user': 'notification', 'text': row.strip()})

    chat_df[['user', 'text']] = chat_df['user_msg'].apply(split_user_msg)
    chat_df.drop('user_msg', axis=1, inplace=True)

    chat_df['date_time'] = pd.to_datetime(chat_df['date_time'], format='%d/%m/%Y, %I:%M %p - ')
    chat_df = chat_df.sort_values('date_time')

    chat_df['date'] = [d.date() for d in chat_df['date_time']]
    chat_df['hour'] = [d.time() for d in chat_df['date_time']]
    chat_df['hour_int'] = chat_df['hour'].apply(lambda x: x.hour)
    chat_df['day'] = chat_df['date_time'].apply(lambda x: x.day_name())
    chat_df['month'] = chat_df['date_time'].apply(lambda x: x.month_name())

    chat_df.drop('date_time', axis=1, inplace=True)

    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    chat_df['month'] = pd.Categorical(chat_df['month'], categories=months_order, ordered=True)
    chat_df['day'] = pd.Categorical(chat_df['day'], categories=days_order, ordered=True)

    return chat_df

#Number of messages
def nb_message(df):
        nb_msg = df.groupby('date')['text'].count().reset_index(name = 'count')
        fig = px.line(nb_msg, x ='date', y ='count',
                      labels = {'date': 'Date','count': 'Number of messages'},
                      title = 'Messages sent per day over a time period')
        fig.update_layout(title_font_size = 24,
                          xaxis_title_font_size = 24,
                          yaxis_title_font_size = 24,
                          xaxis_tickfont_size =16,
                          yaxis_tickfont_size =16)
        return fig

#Top 10 days
def top10days(df):
    top10days = df.groupby('date')['text'].count().sort_values(ascending=False).head(10).reset_index()
    top10days.columns = ['date', 'count']
    #Convert to string
    top10days['date'] = top10days['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
    top10days = top10days.sort_values('count', ascending = False)
    color_scale = px.colors.sequential.Rainbow
    fig = px.bar(top10days, x = 'date', y='count',
                 labels = {"date": "date", "count" : "Number of messages"},
                 title = "Top 10 most active days",
                 color = 'count',
                 color_continuous_scale=color_scale)
    fig.update_layout(title_font_size = 20,
                      xaxis_title_font_size = 14,
                          yaxis_title_font_size = 14,
                          xaxis_tickfont_size =12,
                          yaxis_tickfont_size =12)
    fig.update_xaxes(tickangle = 45)
    fig.update_xaxes(tickmode='array', tickvals=top10days['date'])
    return fig, top10days 


#Monthly activity
def topmonths (df):
    topmonths = df['month'].value_counts().reset_index()
    topmonths.columns = ['month','count']
    topmonths = topmonths.sort_values('month',
                                      ascending = False)
    fig = px.bar(topmonths, x='count',y='month', orientation = 'h',
                  labels = {"month": "month", "count" : "Number of messages"},
                 title = "Chat per month")
    fig.update_layout(title_font_size = 20,
                      xaxis_title_font_size = 14,
                          yaxis_title_font_size = 14,
                          xaxis_tickfont_size =12,
                          yaxis_tickfont_size =12)
    return fig, topmonths 

#Daily activity
def topdays(df):
    topdays = df['day'].value_counts().reset_index()
    topdays.columns = ['day','count']
    topdays = topdays.sort_values('day')
    fig = px.line_polar(topdays, r='count', theta = 'day',
                        line_close = True)
    #Enhance interactivity
    fig.update_traces(fill = 'toself',hoverinfo='all',
                      mode='lines+markers')
    fig.update_layout(polar = dict(radialaxis = dict(visible = True)),
                      showlegend = True,
                      title = {'text' : 'Chat per day of week'})
    topdays = topdays.sort_values('count', ascending = False)
    return fig, topdays

#Daily active time 
def tophours(df):
    tophours = df.groupby('hour_int').size().reset_index(name = "count")
    fig = px.line(tophours, x = 'hour_int', y ='count', 
                  markers = True,
                  labels = {'hour_int': "Hour of the Days",'count' : "Number of messages"},
                  title = "Daily Active Time")
    fig.update_layout(title_font_size = 20,
                      xaxis_title_font_size = 14,
                      yaxis_title_font_size = 14,
                      xaxis_tickfont_size =12,
                      yaxis_tickfont_size =12,
                      xaxis = dict(tickvals = list(range(24))))
    return fig, tophours
#Heat map of month and day
def group_month_day(df):
        grouped_by_month_day = df.groupby(['month', 'day'])['text'].value_counts().reset_index(name = 'count')
        pivot = grouped_by_month_day.pivot_table(index = 'month', columns = 'day',
                                         values = 'count')
        # Ensure the days are ordered in the typical weekly order
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot[ordered_days]
        fig = px.imshow(pivot, color_continuous_scale='Cividis',
                        labels=dict(x="Days of a week", y="Months", color="Activity"),
                        title = "Months and Days activity")
        fig.update_layout(title_font_size = 24,
                          xaxis_title_font_size = 16,
                          yaxis_title_font_size = 16,
                          yaxis_tickfont_size = 16,
                          xaxis_tickangle = 45,
                          margin = dict(t=50,b=50,l=70,r=70),
                          width = 700, height = 400)
        
        return fig,pivot

#Most used words 
nltk.download('stopwords')
custom_stopwords = {'group', 'link', 'invite', 'joined',
                    'message', 'deleted', 'yeah', 'hai',
                    'yes', 'okay', 'ok', 'will', 'use',
                    'using', 'one', 'know', 'guy', 'group',
                    'media', 'omitted'}
stopwords = custom_stopwords.union(set(STOPWORDS)).union(set(nltk.corpus.stopwords.words('english')))
def wordcloud(row):
        text = row.str.lower()
        text_str = ' '.join(text.tolist())
        wordcloud = WordCloud(width = 800, height = 400,
                             random_state = 42, stopwords = stopwords,
                             background_color = 'white').generate(text_str)
        fig, ax = plt.subplots(figsize=(20,10))
        ax.imshow(wordcloud, interpolation ='bilinear')
        ax.axis('off')
        ax.set_title('Wordcloud of the most common words', 
                     fontsize = 30)
        return fig

#Most active users
def top10users(df):
      #Ignore notifications
    df = df[df.user!= 'notification']
      #Ignore phone numbers
    def is_phone_number(s):
            pattern = r"^\+?[\d\s()-]{10,}$"
            return bool(re.match(pattern, s))
    df = df[~df['user'].apply(is_phone_number)]

    top10users = df['user'].value_counts().sort_values(ascending = False).head(10).reset_index()
    top10users.columns = ['user', 'msg_sent']
      #Hide names and get the initials
    def get_initials(name):
        name = re.sub(r'\([^)]*\)', '', name)
        words = name.split()
        initials = ' '.join(word[0].upper() for word in words if word)
        return initials[:3]

    top10users['initials'] = top10users['user'].apply(get_initials)
    top10users.drop('user', axis = 1, inplace =True)

    fig = px.bar(top10users, x = "initials", y = "msg_sent", 
                 labels = {"initials" : "initials", "msg_sent": "messages sent"},
                 title = 'Top 10 users' ) 
    fig.update_layout(title_font_size = 20,
                      xaxis_title_font_size = 14,
                      yaxis_title_font_size = 14,
                      xaxis_tickfont_size = 12,
                      yaxis_tickfont_size = 12)

    return fig, top10users

#Sentiment Analysis
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
def get_sentiment(text):
    scores = sentiments.polarity_scores(text)
    if scores['pos'] > scores['neg'] and scores['pos'] > scores['neu']:
        return 'Positive'
    elif scores['neg'] > scores['pos'] and scores['neg'] > scores['neu']:
        return 'Negative'
    else :
        return 'Neutral'
def sentiment_score(df):
    df['Positive'] = [sentiments.polarity_scores(i)['pos']
                      for i in df['text']]
    df['Negative'] = [sentiments.polarity_scores(i)['neg']
                     for i in df['text']]
    df['Neutral'] = [sentiments.polarity_scores(i)['neu']
                    for i in df['text']]
    df['Sentiment'] = df['text'].apply(get_sentiment)

    mask = df['user'] =='notification'
    df.loc[mask, ['Positive', 'Negative', 'Neutral', 'Sentiment']] = 'NA'
    return df
def chat_sentiment(df):
    df = sentiment_score(df)
    chat_filtered = df[df['Sentiment']!='NA']
    chat_sentiment = chat_filtered['Sentiment'].value_counts().sort_values().reset_index()
    chat_sentiment.columns = ['sentiment', 'count']
    fig = px.pie(chat_sentiment, names = 'sentiment', values = "count",
                  labels = {"sentiment": "Sentiment", "count": "Number of messages"},
                  title = "Chat sentiment distribution")
    fig.update_traces(textinfo = "label+percent", 
                      textfont_size = 20)
    fig.update_layout(title_font_size = 20,
                      legend_title_font_size = 10,
                      legend_font_size = 12,
                      margin = dict(t=100,b=70,l=70,r=100),
                      height = 500, width = 700)
    return fig

#Most used emojis
def extract_emojis(text):
    emoji_list = [char for char in text if char in emoji.EMOJI_DATA]
    return emoji_list
def get_description(emoji_char):
    return emoji.EMOJI_DATA[emoji_char]['en']

  #Count the extracted emojis
def top_emojis(df):
    def count(df):
        all_emojis = []
        for text in df['text']:
            all_emojis.extend(extract_emojis(text))
        emoji_counts = Counter(all_emojis)
        return emoji_counts
    emoji_counts = count(df)

    #Convert to dataframe
    emoji_df = pd.DataFrame.from_dict(emoji_counts, orient = 'index',
                                          columns = ['count'])
    emoji_df = emoji_df.reset_index()
    emoji_df = emoji_df.rename(columns = {'index' : 'emoji'})

      #Add description and plot
    emoji_df['description'] = emoji_df['emoji'].apply(get_description)
    emoji_df['emoji_description'] = emoji_df['emoji']+' - ' + emoji_df['description']
    emoji_df.drop('description', axis = 1, inplace = True)
    top10emojis = emoji_df.sort_values('count', ascending = False).head(10)
    fig = px.pie(top10emojis, names='emoji_description',values = 'count',
                 title = 'Top 10 Emojis', hole = 0.3)
    fig.update_traces(hovertemplate='%{label}Count: %{value}')
    return fig, top10emojis




def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = working_dir
    files = get_file_list(folder_path)
    selected_file = st.selectbox('Select a file', files)

    if selected_file:
        st.write(f"Selected file: {selected_file}")
        file_path = os.path.join(folder_path, selected_file)
        chat_df = preprocess(file_path)

        if chat_df is not None:
            st.subheader('Dataset Overview')
            st.write(f"The dataset contains {chat_df.shape[0]:,} rows and {chat_df.shape[1]} columns."  )
            st.subheader("Sample of 10 messages from the chat:")
            st.write(chat_df.sample(10))
            plot_options = ["Messages Sent per Day",
                            "Top 10 Most Active Days",
                             "Chat per month",
                              "Chat per day of week",
                              "Daily active time",
                              "Heat map of monthly activity and daily activity",
                              "Most used words",
                              "Most active users",
                              "Sentiment Analysis",
                              "Top 10 emojis"]
            plot_option = st.selectbox("Choose a plot to display",
                                       options = plot_options+['None'],
                                       index = None)
            

            #Messages sent per day
            if plot_option =="Messages Sent per Day":
                fig = nb_message(chat_df)
                st.plotly_chart(fig)  #display the plot on streamlit
            #Top 10 days
            elif plot_option=="Top 10 Most Active Days":
                col1, col2 = st.columns([2,1])
                with col1:
                    fig, top10 = top10days(chat_df) #function returns top10
                    st.plotly_chart(fig) #display the plot on streamlit
                with col2:
                    st.write(top10)
            #Monthly activity
            elif plot_option == "Chat per month":
                col1, col2 = st.columns([2,1])
                with col1:
                    fig, top12 = topmonths(chat_df)
                    st.plotly_chart(fig) #display the plot on streamlit
                with col2:
                    st.write(top12.sort_values('count', 
                                           ascending = False))
            #Daily activity
            elif plot_option =="Chat per day of week":
                col1,col2 = st.columns([2,1])
                with col1:
                    fig, top7 = topdays(chat_df)
                    st.plotly_chart(fig)
                with col2:
                    st.write(top7)

            #Daily active time 
            elif plot_option == "Daily active time":
                fig, top24 = tophours(chat_df)
                st.plotly_chart(fig)

            #Heat map of monthly activity and daily activity
            elif plot_option =="Heat map of monthly activity and daily activity":
                col1, col2 = st.columns([2,1])
                with col1 :
                    fig, pivot = group_month_day(chat_df)
                    st.plotly_chart(fig)
                with col2:
                    inference_statement = (
                        "\n\n"
                        "The intensity of group chat activity is represented by color: lighter shades indicating higher levels of engagement.\n\n"
                        "A consistent pattern shows the group is more active on weekends throughout all months.\n\n"
                        "September stands out as the most active month in the chat: this month has the most lighter blue shade and more yellow gradients."
                    )
                    st.write(inference_statement)

            #Most used words
            elif plot_option=="Most used words":
                fig = wordcloud(chat_df['text'])
                st.pyplot(fig)

            #Most active users
            elif plot_option=="Most active users":
                 col1, col2 = st.columns([2,1])
                 with col1:   
                    fig, topusers = top10users(chat_df)
                    st.plotly_chart(fig)
                 with col2:
                    st.write(topusers)

            #Sentiment Analysis
            elif plot_option=="Sentiment Analysis":
                col1, col2 = st.columns([2,1])
                with col1 :
                    fig = chat_sentiment(chat_df)
                    df = sentiment_score(chat_df)
                    st.plotly_chart(fig)
                with col2 :
                    df = df.sample(15)
                    st.write(df[['text', 'Sentiment']])
            
            #Top 10 emojis
            elif plot_option=="Top 10 emojis":
                fig, topemojis = top_emojis(chat_df)
                st.plotly_chart(fig)
                st.write(topemojis)

            else :
                None

if __name__ == "__main__":
    main()