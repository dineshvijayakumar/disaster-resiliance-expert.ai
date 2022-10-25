# -*- coding: utf-8 -*-

# !pip install expertai-nlapi

# !pip install snscrape

from flask import Flask, request
import warnings
import time
import requests
from pandas.io.json import json_normalize
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import json
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from jinja2 import Template
from datetime import datetime, date, timedelta
import csv

from expertai.nlapi.cloud.client import ExpertAiClient
client = ExpertAiClient()


app = Flask(__name__)
warnings.simplefilter('ignore')

os.environ["EAI_USERNAME"] = 'dinesh.vijayakumar@live.com'
os.environ["EAI_PASSWORD"] = 'thXNs6X!7Mg3'

base_path = "/"
domain = "builderprogram-dvijayakumar.quickbase.com"
usertoken = "bz64sg_m2ax_0_b3v5uf6fjm5jq3nrwnkn7ixap"
headers = {'QB-Realm-Hostname': domain, 'User-Agent': 'PythonUtility',
           'Authorization': 'QB-USER-TOKEN ' + usertoken}
disastersTableid = "bsq4ndg9r"
disastersReportid = "5"
perBatchRecordsCount = 5000
filename = "Disaster Events.csv"

tweetsTableid="bsq4namtn"
entitiesTableid="bssdwittw"
behaviorsTableid="bssdx6irn"
emotionsTableid="bssdx7eu6"

tweetsClist="8.6.9.10.24.20.21.22.23.14.7.17"
entitiesClist="8.6.7"
behaviorsClist="9.6.7"
emotionsClist="8.6.7"


@app.route('/')
def home():

    warnings.simplefilter('ignore')
   
    disasterdf = exportqbdata()

    twitter_handles=[]
    tweets_df=pd.DataFrame({})
    entities_df=pd.DataFrame({})
    behavioral_traits_df=pd.DataFrame({})
    emotional_traits_df=pd.DataFrame({})

    for index,row in disasterdf.iterrows():
        twitter_hashtags=row["Combined Hashtag"]
        tweet_loc=row["Near Location"]
        twitter_duration_days=row["Monitor Duration Days (backwards from today)"]
        #event_tweets_df,event_entity_df,event_behavioral_traits_df, event_emotional_traits_df=scrapeTweets(twitter_hashtags,tweet_loc,twitter_duration_days)
        event_tweets_df=scrapeTweets(twitter_hashtags,tweet_loc,twitter_duration_days)
        event_tweets_df["Related Disaster"]=row["Record ID#"]
        tweets_df=tweets_df.append(event_tweets_df)
    
    tweets_entities_list=[]
    tweets_emotional_traits_list=[]
    tweets_behavioral_traits_list=[]

    tweet_entities_df=pd.DataFrame({})
    tweet_emotional_traits_df=pd.DataFrame({})
    tweet_behavioral_traits_df=pd.DataFrame({})
    tweets_df["lang"]=""
    tweets_df[["retweet_count","reply_count","like_count","quote_count"]]=0
    for i,row in tweets_df.iterrows():
        temp_tweet_meta_data_list=getTweetByID(row['Tweet Id'])
        tweets_df.at[i,"lang"]=temp_tweet_meta_data_list[0]
        tweets_df.at[i,"retweet_count"]=temp_tweet_meta_data_list[1]
        tweets_df.at[i,"reply_count"]=temp_tweet_meta_data_list[2]
        tweets_df.at[i,"like_count"]=temp_tweet_meta_data_list[3]
        tweets_df.at[i,"quote_count"]=temp_tweet_meta_data_list[4]
        tweets_df.at[i,"Sentiment"]=findSentiment(row["Text"],temp_tweet_meta_data_list[0])
     
    for i,row in tweets_df.iterrows():
        temp_entities=findEntities(row["Tweet Id"],row["Text"],row["lang"])
        if len(temp_entities)!=0:
            tweets_entities_list.extend(temp_entities)

        temp_emotional_traits=findEmotionalTraits(row["Tweet Id"],row["Text"],row["lang"])
        if len(temp_emotional_traits)!=0:
            tweets_emotional_traits_list.extend(temp_emotional_traits)

        temp_behavioral_traits=findBehavioralTraits(row["Tweet Id"],row["Text"],row["lang"])
        if len(temp_behavioral_traits)!=0:
            tweets_behavioral_traits_list.extend(temp_behavioral_traits)


    tweet_entities_df=tweet_entities_df.append(pd.DataFrame(tweets_entities_list,columns=["Related Tweet","Entity","Entity Type"]))
    tweet_behavioral_traits_df=tweet_behavioral_traits_df.append(pd.DataFrame(tweets_behavioral_traits_list,columns=["Related Tweet","Behavior","Frequency"]))
    tweet_emotional_traits_df=tweet_emotional_traits_df.append(pd.DataFrame(tweets_emotional_traits_list,columns=["Related Tweet","Emotion","Frequency"]))
    
    tweet_entities_df=tweet_entities_df[tweet_entities_df.duplicated()==False]
    tweet_behavioral_traits_df=tweet_behavioral_traits_df[tweet_behavioral_traits_df.duplicated()==False]
    tweet_emotional_traits_df=tweet_emotional_traits_df[tweet_emotional_traits_df.duplicated()==False]
    print(str(tweet_entities_df.shape[0])+" entity records found...")
    print(str(tweet_behavioral_traits_df.shape[0])+" emotional trait records found...")
    print(str(tweet_emotional_traits_df.shape[0])+" behavioral trait records found...")

    print(tweets_df.head())
    print(tweet_entities_df.head())
    print(tweet_behavioral_traits_df.head())
    print(tweet_emotional_traits_df.head())

    tweets_df["Datetime_MS_QB"]=tweets_df["Datetime"].astype("str").str.replace(" ","T")
    tweets_df["Datetime_IST_QB"]=pd.DatetimeIndex(tweets_df["Datetime"]).tz_convert('Asia/Kolkata').astype(str).str.replace("\+05:30","").str.replace(" ","T")
    tweets_df.drop(columns=["Datetime"],inplace=True)

    importTweetsStatus,tweetsMetaData=pushTweetsToQB(tweets_df,domain,usertoken,tweetsTableid,tweetsClist)
    importEntitiesStatus,entitiesMetaData=pushTweetsToQB(tweet_entities_df,domain,usertoken,entitiesTableid,entitiesClist)
    importBehaviorsStatus,behaviorsMetaData=pushTweetsToQB(tweet_behavioral_traits_df,domain,usertoken,behaviorsTableid,behaviorsClist)
    importEmotionsStatus,emotionsMetaData=pushTweetsToQB(tweet_emotional_traits_df,domain,usertoken,emotionsTableid,emotionsClist)

    print('Here in last running from app')

    Success_Response = '{"Succes": true, "message": "Completed"}'
    response_dict = json.loads(Success_Response)
    print(response_dict)
    return response_dict


def importDataToQB(importdata, tableid, headers):
    url = 'https://api.quickbase.com/v1/records'
    queryBody = {
        "to": tableid,
        "data": importdata,
        "fieldToReturn": [
            3
        ]
    }

    rQuery = requests.post(url,
                           headers=headers,
                           json=queryBody
                           )
    importStatus = ""
    metaData = ""
    if rQuery.status_code == 200:
        recordsProcessed = json.loads(
            rQuery.text)['metadata']['totalNumberOfRecordsProcessed']
        print("Number of records processed..."+str(recordsProcessed))
        importStatus = "Success"
        metaData = recordsProcessed
    else:
        print(rQuery.text)
        importStatus = "Failed"
        metaData = rQuery.text
    return importStatus, metaData


def pushTweetsToQB(df,domain,usertoken,tableid,clist):

    importdatadf=df.copy(deep=True)
    headers = {'QB-Realm-Hostname': domain,'User-Agent': 'PythonUtility','Authorization': 'QB-USER-TOKEN '+usertoken}
    perBatchRecordsCount=10000


    rowsCount=importdatadf.shape[0]
    print("Records to be imported..."+str(rowsCount))
    print(importdatadf.head())

    importdatadf=importdatadf.fillna(value="")
    if rowsCount>perBatchRecordsCount:
        for skip in range(0,rowsCount,perBatchRecordsCount):
            importsubdata=importdatadf.loc[skip:skip+perBatchRecordsCount]
            importdatadf[skip:].to_csv("temp1.csv")
            importsubdata.columns=list(clist.split(sep="."))
            importdataJSON=json.loads(importsubdata.to_json(orient='records'))
    else:
        importsubdata=importdatadf
        importsubdata.columns=list(clist.split(sep="."))
        importdataJSON=json.loads(importsubdata.to_json(orient='records'))
    for index,item in enumerate(importdataJSON):
        for key in item.keys():
            importdataJSON[index][key]=dict({"value":importdataJSON[index][key]})
    importStatus, metaData=importDataToQB(importdataJSON,tableid,headers)
    return importStatus, metaData

def getTweetByID(id):
    search_url = "https://api.twitter.com/2/tweets"
    bearer_token="AAAAAAAAAAAAAAAAAAAAAOvvRQEAAAAAIYR6XMy63KCVbWaN2VdBqv76FgI%3DYiwwZdh0h1OC1W6ka5lGOGeShWecq878kytDtdzUBWoageLNWU"
    query_params =  {'ids':id,     
                  'tweet.fields':'created_at,lang,text,geo,author_id,id,public_metrics,referenced_tweets',
                  'expansions':'geo.place_id,author_id', 
                  'place.fields':'contained_within,country,country_code,full_name,geo,id,name,place_type',
                  'user.fields':'description,username,id'}
    #                'max_results':'500'}
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    response = requests.request("GET", search_url, headers=headers, params=query_params)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    json_response= response.json()
    df = pd.json_normalize(json_response['data'])
    return [df["lang"][0],df["public_metrics.retweet_count"][0],df["public_metrics.reply_count"][0],df["public_metrics.like_count"][0],df["public_metrics.quote_count"][0]]

def findSentiment(x,lang):
    text=x
    language= lang
    try:
        output = client.specific_resource_analysis(
          body={"document": {"text": text}}, 
          params={'language': language, 'resource': 'sentiment'})
        return output.sentiment.overall
    except:
        return 0
    
def findEntities(id,x,lang):
    text =x
    language= lang
    entities=[]
    try:
        output = client.specific_resource_analysis(
          body={"document": {"text": text}}, 
          params={'language': language, 'resource': 'entities'})
        for entity in output.entities:
          entities.append([id,entity.lemma,entity.type_])
        return entities
    except:
        return entities
    
def findEmotionalTraits(id,x,lang):
    taxonomy='emotional-traits'
    text =x
    language= lang
    emotions=[]
    try:
        document = client.classification(body={"document": {"text": text}}, params={'taxonomy': taxonomy,'language': language})
        for category in document.categories:
            emotions.append([id,category.label,category.frequency])
        return emotions
    except:
        return emotions

def findBehavioralTraits(id,x,lang):
    taxonomy='behavioral-traits'
    text =x
    language= lang
    behaviors=[]
    try:
        document = client.classification(body={"document": {"text": text}}, params={'taxonomy': taxonomy,'language': language})
        for category in document.categories:
            behaviors.append([id,category.label,category.frequency])
        return behaviors
    except:
        return behaviors

def getQBBatchDF(batchRecordCount, skipStart, firstIter):
    print("Generating the records from " + str(int(skipStart) + 1) + "...")
    url = 'https://api.quickbase.com/v1/reports/' + disastersReportid + '/run?tableId=' + disastersTableid + '&skip=' + str(
        skipStart) + '&top=' + str(batchRecordCount)
    queryBody = {
    }

    retryCount = 0
    qbdataDF = pd.DataFrame()
    qbrecordsDict = dict()
    qbfieldsDict = []

    try:
        print(url)
        rQuery = requests.post(url,
                               headers=headers,
                               json=queryBody,
                               verify=False
                               )

        if rQuery.status_code == 200:
            responseQuery = json.loads(json.dumps(rQuery.json(), indent=4))
            qbfieldsDict = list(responseQuery["fields"])
            qbrecordsDict = (responseQuery["data"])
            qbdataDF = pd.DataFrame.from_dict(qbrecordsDict, orient="columns")
            for col in qbdataDF.columns:
                qbdataDF[col] = pd.json_normalize(qbdataDF[col], max_level=0)
        else:
            print("Skipped the records from " + str(int(skipStart) + 1) + " to " + str(
                int(skipStart) + perBatchRecordsCount) + "...")

        return qbdataDF, qbfieldsDict, rQuery.status_code
    except IndexError as e:
        print(e)
        while retryCount < 3:
            retryCount += 1
            print("Retry "+str(retryCount))
            time.sleep(5)
            continue


def exportqbdata(skipStart=0, batchRecordCount=0):
    if batchRecordCount == 0:
        batchRecordCount = perBatchRecordsCount
    if (batchRecordCount <= perBatchRecordsCount):
        outputdf, qbfieldsDict, statusCode = getQBBatchDF(
            batchRecordCount, skipStart, firstIter=True)
        outputdf.to_csv(filename, encoding="utf-8-sig")
    else:
        outputdf = pd.DataFrame()
        lastbatch = False
        skipStartIter = int(skipStart)
        firstIter = True
        print(batchRecordCount, skipStart)
        while lastbatch == False:
            qbbatchdf, qbfieldsDict, statusCode = getQBBatchDF(
                perBatchRecordsCount, skipStartIter, firstIter)
            print(statusCode)
            if statusCode == 200:
                firstIter = False
                outputdf = outputdf.append(qbbatchdf)
                outputdf.to_csv("Disaster Events.csv", encoding="utf-8-sig")
                skipStartIter += perBatchRecordsCount
                if skipStartIter > (int(batchRecordCount)+int(skipStart)):
                    lastbatch = True
            else:
                continue

    qbfieldsDF = pd.DataFrame.from_dict(qbfieldsDict)
    qbfieldsDF.set_index("id", inplace=True)
    qbcols = outputdf.columns
    qbcolnames = []
    for col in qbcols:
        qbcolnames.append(qbfieldsDF["label"][int(col)])
    outputdf.columns = qbcolnames
    outputdf.reset_index(drop=True, inplace=True)
    print(outputdf.shape)
    outputdf.to_csv(filename, encoding="utf-8-sig")
    print("Exporting "+str(batchRecordCount)+" records to a CSV file...")
    return outputdf


# Initiate webscrapping of tweets
def scrapeTweets(twitter_hashtags,tweet_loc,twitter_duration_days): # Initiate webscrapping of tweets
    tweets_list = []
    tweets_df=pd.DataFrame({})

    for i in range(len(twitter_hashtags)):
        
        print(twitter_duration_days)

        batches=[]
        batchesTimestamp=[]

        end_date=datetime.today()+timedelta(days=1)
        end_date_temp=end_date
        duration_days_temp=twitter_duration_days+1

        while duration_days_temp>=0:
            end_date_temp=end_date-timedelta(days=duration_days_temp)
            batches.append(datetime.strftime(end_date_temp,"%Y-%m-%d"))
            batchesTimestamp.append(end_date_temp)
            duration_days_temp=duration_days_temp-999

        print(batches)

        end_date_temp=datetime.strftime(end_date,"%Y-%m-%d")

        for k in reversed(range(0,len(batches))):
            print("Retrieving tweets between "+batches[k]+" and "+end_date_temp+" ("+str((datetime.strptime(end_date_temp,"%Y-%m-%d")-datetime.strptime(batches[k],"%Y-%m-%d")).days)+" days)")
            for tweet in sntwitter.TwitterSearchScraper(twitter_hashtags[i]+' near:"'+tweet_loc+'" within:50km since:'+batches[k]+' until:'+end_date_temp).get_items():
                tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username])
            
            temp_df=pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
            tweets_df=tweets_df.append(temp_df)
            end_date_temp=batches[k]
        
    tweets_df=tweets_df[tweets_df.duplicated()==False]

    print(str(tweets_df.shape[0])+" tweets fetched...")

    return tweets_df


def importDataToQB(importdata, tableid, headers):
    url = 'https://api.quickbase.com/v1/records'
    queryBody = {
        "to": tableid,
        "data": importdata,
        "fieldToReturn": [
            3
        ]
    }

    rQuery = requests.post(url,
                           headers=headers,
                           json=queryBody
                           )
    importStatus = ""
    metaData = ""
    if rQuery.status_code == 200:
        recordsProcessed = json.loads(
            rQuery.text)['metadata']['totalNumberOfRecordsProcessed']
        print("Number of records processed..."+str(recordsProcessed))
        importStatus = "Success"
        metaData = recordsProcessed
    else:
        print(rQuery.text)
        importStatus = "Failed"
        metaData = rQuery.text
    return importStatus, metaData


def pushTweetsToQB(df,domain,usertoken,tableid,clist):

    importdatadf=df.copy(deep=True)
    headers = {'QB-Realm-Hostname': domain,'User-Agent': 'PythonUtility','Authorization': 'QB-USER-TOKEN '+usertoken}
    perBatchRecordsCount=10000


    rowsCount=importdatadf.shape[0]
    print("Records to be imported..."+str(rowsCount))
    print(importdatadf.head())

    importdatadf=importdatadf.fillna(value="")
    if rowsCount>perBatchRecordsCount:
        for skip in range(0,rowsCount,perBatchRecordsCount):
            importsubdata=importdatadf.loc[skip:skip+perBatchRecordsCount]
            importdatadf[skip:].to_csv("temp1.csv")
            importsubdata.columns=list(clist.split(sep="."))
            importdataJSON=json.loads(importsubdata.to_json(orient='records'))
    else:
        importsubdata=importdatadf
        importsubdata.columns=list(clist.split(sep="."))
        importdataJSON=json.loads(importsubdata.to_json(orient='records'))
    for index,item in enumerate(importdataJSON):
        for key in item.keys():
            importdataJSON[index][key]=dict({"value":importdataJSON[index][key]})
    importStatus, metaData=importDataToQB(importdataJSON,tableid,headers)
    return importStatus, metaData


