# -*- coding: utf-8 -*-
"""Intelligent Disaster Issues Redressal System using expertAI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xk369rTTBDrdtpeDby3FnP_6Aah8OIYc
"""

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

# from google.colab import drive
# drive.mount('/content/drive')
# base_path="/content/drive/MyDrive/AIML/Hackathon_Files/Devpost/expert.ai 2022/"
base_path = "/"
domain = "builderprogram-dvijayakumar.quickbase.com"
usertoken = "bz64sg_m2ax_0_b3v5uf6fjm5jq3nrwnkn7ixap"
headers = {'QB-Realm-Hostname': domain, 'User-Agent': 'PythonUtility',
           'Authorization': 'QB-USER-TOKEN ' + usertoken}
tableid = "bsq4ndg9r"
reportid = "5"
perBatchRecordsCount = 5000
filename = "Disaster Events.csv"


@app.route('/')
def home():

    warnings.simplefilter('ignore')

    # headers = {'QB-Realm-Hostname': 'builderprogram-dvijayakumar.quickbase.com','User-Agent': 'raspberry-pi','Authorization': 'QB-USER-TOKEN bz64sg_m2ax_dfuzmdhdke9qp6bf6zn2db9vtsep'}

    disasterdf = exportqbdata()

    twitter_handles = []
    tweets_df = pd.DataFrame({})
    for index, row in disasterdf.iterrows():
        twitter_hashtags = row["Combined Hashtag"]
        tweet_loc = row["Near Location"]
        twitter_duration_days = row["Monitor Duration Days (backwards from today)"]
        print('here 194')
        print(twitter_hashtags, tweet_loc, twitter_duration_days)
        tweets_df = tweets_df.append(scrapeTweets(
            twitter_hashtags, tweet_loc, twitter_duration_days))
        tweets_df = tweets_df[(tweets_df["Datetime"].dt.year == pd.Timestamp('now').year) & (
            tweets_df["Datetime"].dt.month == pd.Timestamp('now').month) & (tweets_df["Datetime"].dt.day >= (pd.Timestamp('now').day-1))]
        tweets_df["Sentiment"] = tweets_df["Text"].map(
            lambda x: findSentiment(x))
        importStatus, metaData = pushTweetsToQB(
            tweets_df, twitter_duration_days)
        importStatus, metaData
    tweets_df.head()

    pd.Timestamp('today').day-1

    importStatus, metaData = pushTweetsToQB(tweets_df, twitter_duration_days)

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


def pushTweetsToQB(df, twitter_duration_days):
    #domain=input("Enter the quickbase domain name as in <domain>.quickbase.com...")
    domain = "builderprogram-dvijayakumar.quickbase.com"

    #usertoken=input("Enter user token...")
    usertoken = "bz64sg_m2ax_0_b3v5uf6fjm5jq3nrwnkn7ixap"

    #tableid=input("Enter the Table ID for data export...")
    tableid = "bsq4namtn"

    #clist=input("Enter the clist for data import...")
    clist = "8.6.9.14.7.17"
    #importFilePath=base_path+"Tweets DataFrame 20221005_1.csv"
    end_date = datetime.today()
    start_date = end_date-timedelta(days=twitter_duration_days)
    importFilePath = base_path+"QB Dataframe" + str(datetime.strftime(
        start_date, "%Y-%m-%d"))+"_to_"+str(datetime.strftime(end_date, "%Y-%m-%d"))+".csv"
    importdatadf = df.copy(deep=True)
    # importdatadf=pd.read_csv(importFilePath,encoding="utf-8")
    importdatadf.to_csv(importFilePath, encoding='utf-8')
    headers = {'QB-Realm-Hostname': domain, 'User-Agent': 'PythonUtility',
               'Authorization': 'QB-USER-TOKEN '+usertoken}
    perBatchRecordsCount = 10000

    # importdatadf=pd.read_csv(importFilePath,encoding="utf-8")

    importdatadf["Datetime_MS_QB"] = importdatadf["Datetime"].astype(
        "str").str.replace(" ", "T")
    importdatadf["Datetime_IST_QB"] = pd.DatetimeIndex(importdatadf["Datetime"]).tz_convert(
        'Asia/Kolkata').astype(str).str.replace("\+05:30", "").str.replace(" ", "T")

    importdatadf.drop(columns=["Datetime"], inplace=True)
    rowsCount = importdatadf.shape[0]
    print("Records to be imported..."+str(rowsCount))
    print(importdatadf.head())

    importdatadf = importdatadf.fillna(value="")
    if rowsCount > perBatchRecordsCount:
        for skip in range(0, rowsCount, perBatchRecordsCount):
            importsubdata = importdatadf.loc[skip:skip+perBatchRecordsCount]
            importdatadf[skip:].to_csv("temp1.csv")
            importsubdata.columns = list(clist.split(sep="."))
            importdataJSON = json.loads(
                importsubdata.to_json(orient='records'))
            # for index,item in enumerate(importdataJSON):
            #   for key in item.keys():
            # importdataJSON[index][key]="{\"value\":"+str(importdataJSON[index][key])+"\"}"
            #      importdataJSON[index][key]=dict({"value":importdataJSON[index][key]})
            # print(importdataJSON)
            # break
            # importDataToQB(importdataJSON,tableid,headers)
    else:
        importsubdata = importdatadf
        # print(clist)
        importsubdata.columns = list(clist.split(sep="."))
        importdataJSON = json.loads(importsubdata.to_json(orient='records'))
        # print(importdataJSON)
    for index, item in enumerate(importdataJSON):
        for key in item.keys():
            # importdataJSON[index][key]="{\"value\":"+str(importdataJSON[index][key])+"\"}"
            importdataJSON[index][key] = dict(
                {"value": importdataJSON[index][key]})
    importStatus, metaData = importDataToQB(importdataJSON, tableid, headers)
    return importStatus, metaData


def findSentiment(x):
    text = x
    language = 'en'
    output = client.specific_resource_analysis(
        body={"document": {"text": text}},
        params={'language': language, 'resource': 'sentiment'})
    return output.sentiment.overall


def getQBBatchDF(batchRecordCount, skipStart, firstIter):
    print("Generating the records from " + str(int(skipStart) + 1) + "...")
    url = 'https://api.quickbase.com/v1/reports/' + reportid + '/run?tableId=' + tableid + '&skip=' + str(
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
                outputdf.to_csv("Catalog_Residents.csv", encoding="utf-8-sig")
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
def scrapeTweets(twitter_hashtags, tweet_loc, twitter_duration_days):
    tweets_list = []

    tweets_df = pd.DataFrame({})

    for i in range(len(twitter_hashtags)):

        print(twitter_duration_days)

        batches = []
        batchesTimestamp = []

        end_date = datetime.today()+timedelta(days=1)
        end_date_temp = end_date
        duration_days_temp = twitter_duration_days+1

        while duration_days_temp >= 0:
            end_date_temp = end_date-timedelta(days=duration_days_temp)
            batches.append(datetime.strftime(end_date_temp, "%Y-%m-%d"))
            batchesTimestamp.append(end_date_temp)
            duration_days_temp = duration_days_temp-999

        print(batches)

        end_date_temp = datetime.strftime(end_date, "%Y-%m-%d")

        for k in reversed(range(0, len(batches))):
            print("Retrieving tweets between "+batches[k]+" and "+end_date_temp+" ("+str((datetime.strptime(
                end_date_temp, "%Y-%m-%d")-datetime.strptime(batches[k], "%Y-%m-%d")).days)+" days)")
            for tweet in sntwitter.TwitterSearchScraper(twitter_hashtags[i]+' near:"'+tweet_loc+'" within:50km since:'+batches[k]+' until:'+end_date_temp).get_items():
                tweets_list.append(
                    [tweet.date, tweet.id, tweet.content, tweet.username])
            temp_df = pd.DataFrame(tweets_list, columns=[
                                   'Datetime', 'Tweet Id', 'Text', 'Username'])
            tweets_df = tweets_df.append(temp_df)
            end_date_temp = batches[k]

        tweets_df = tweets_df[tweets_df.duplicated() == False]
        #tweets_df["Datetime_IST"]=pd.DatetimeIndex(pd.to_datetime(tweets_df["Datetime"], unit='ms')).tz_convert('Asia/Kolkata')
        # tweets_df.to_json(base_path+"tweets_"+twitter_hashtags[i]+"_"+str(datetime.strftime(batchesTimestamp[0],"%Y-%m-%d"))+"_to_"+str(datetime.strftime(end_date-timedelta(days=1),"%Y-%m-%d"))+".json")
    return tweets_df

# for days in twitter_duration_days:
#   #endDate=startDate+timedelta(days=days)
#   #startDate=datetime(2022,10,3)
#   end_date=datetime.today()
#   start_date=end_date-timedelta(days=days)
#   hurricaneIan_jsonFileName=base_path+"tweets_"+twitter_hashtags[0]+"_"+str(datetime.strftime(start_date,"%Y-%m-%d"))+"_to_"+str(datetime.strftime(end_date,"%Y-%m-%d"))+".json"
#   print(hurricaneIan_jsonFileName)
#   tweets_df=pd.read_json(hurricaneIan_jsonFileName,orient = 'records')


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


def pushTweetsToQB(df, twitter_duration_days):
    #domain=input("Enter the quickbase domain name as in <domain>.quickbase.com...")
    domain = "builderprogram-dvijayakumar.quickbase.com"

    #usertoken=input("Enter user token...")
    usertoken = "bz64sg_m2ax_0_b3v5uf6fjm5jq3nrwnkn7ixap"

    #tableid=input("Enter the Table ID for data export...")
    tableid = "bsq4namtn"

    #clist=input("Enter the clist for data import...")
    clist = "8.6.9.14.7.17"
    #importFilePath=base_path+"Tweets DataFrame 20221005_1.csv"
    end_date = datetime.today()
    start_date = end_date-timedelta(days=twitter_duration_days)
    importFilePath = base_path+"QB Dataframe" + str(datetime.strftime(
        start_date, "%Y-%m-%d"))+"_to_"+str(datetime.strftime(end_date, "%Y-%m-%d"))+".csv"
    importdatadf = df.copy(deep=True)
    # importdatadf=pd.read_csv(importFilePath,encoding="utf-8")
    importdatadf.to_csv(importFilePath, encoding='utf-8')
    headers = {'QB-Realm-Hostname': domain, 'User-Agent': 'PythonUtility',
               'Authorization': 'QB-USER-TOKEN '+usertoken}
    perBatchRecordsCount = 10000

    # importdatadf=pd.read_csv(importFilePath,encoding="utf-8")

    importdatadf["Datetime_MS_QB"] = importdatadf["Datetime"].astype(
        "str").str.replace(" ", "T")
    importdatadf["Datetime_IST_QB"] = pd.DatetimeIndex(importdatadf["Datetime"]).tz_convert(
        'Asia/Kolkata').astype(str).str.replace("\+05:30", "").str.replace(" ", "T")

    importdatadf.drop(columns=["Datetime"], inplace=True)
    rowsCount = importdatadf.shape[0]
    print("Records to be imported..."+str(rowsCount))
    print(importdatadf.head())

    importdatadf = importdatadf.fillna(value="")
    if rowsCount > perBatchRecordsCount:
        for skip in range(0, rowsCount, perBatchRecordsCount):
            importsubdata = importdatadf.loc[skip:skip+perBatchRecordsCount]
            importdatadf[skip:].to_csv("temp1.csv")
            importsubdata.columns = list(clist.split(sep="."))
            importdataJSON = json.loads(
                importsubdata.to_json(orient='records'))
            # for index,item in enumerate(importdataJSON):
            #   for key in item.keys():
            # importdataJSON[index][key]="{\"value\":"+str(importdataJSON[index][key])+"\"}"
            #      importdataJSON[index][key]=dict({"value":importdataJSON[index][key]})
            # print(importdataJSON)
            # break
            # importDataToQB(importdataJSON,tableid,headers)
    else:
        importsubdata = importdatadf
        # print(clist)
        importsubdata.columns = list(clist.split(sep="."))
        importdataJSON = json.loads(importsubdata.to_json(orient='records'))
        # print(importdataJSON)
    for index, item in enumerate(importdataJSON):
        for key in item.keys():
            # importdataJSON[index][key]="{\"value\":"+str(importdataJSON[index][key])+"\"}"
            importdataJSON[index][key] = dict(
                {"value": importdataJSON[index][key]})
    importStatus, metaData = importDataToQB(importdataJSON, tableid, headers)
    return importStatus, metaData
