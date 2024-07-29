# Whatsapp Chat Data Analysis

## Overview 

This [notebook](whatsapp_chat_da.ipynb) involves a comprehensive data analysis of a WhatsApp chat dataset. The analysis covers various aspects such as preprocessing, message activity, sentiment analysis, and visualization. The key goals are to understand chat patterns, identify active periods, and analyze sentiment and emoji usage.

## Project Structure

  1. **Preprocessing** : The raw text file is cleaned and converted into a structured format. The cleaned data is organized into a pandas DataFrame with the following columns : "user", "text", "date", "hour", "hour_int", "day", "month".
  2. **Analysis** :
     - Messages sent per day : Count the number of messages sent each day throughout the period covered by the dataset.
     - Top 10 most active days : Identify the 10 days with the highest message activity.
     - Activity per month : Analyze message activity trends on a monthly basis.
     - Most active day of the week : Determine which day of the week has the highest message count.
     - Most active time of day : Find out the peak times for message activity during the day.
     - Heat map : Show the activity across months and days, revealing that the group is more active on weekends and highlighting September as the most active month.
     - Word Cloud : Displays the most frequently used words in the chat.
     - Top 10 most active users : Identify the initials of the top 10 users based on message count.
     - Sentiment Analysis : Perform sentiment analysis to determine that the chat is predominantly neutral.
     - Top 10 emojis : Count the usage of the 10 most commonly used emojis in the chat.
  3. **Deployment** : The project is deployed and available for interactive exploration via Streamlit. You can access the deployed app using this [link](https://f9fd9aw62eksd6yzhn7xhn.streamlit.app/?fbclid=IwY2xjawEUsPFleHRuA2FlbQIxMAABHVftEMw6biw5vT_zSYCpwjWiKUM-wTqSG-j7oEX2M3EH2psqWhuBNPJ91A_aem_LSPd9zDbkvIXFcakZBZNIQ)
