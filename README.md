## Inspiration

Natural disasters have a major effect on the livelihood and mental state of disaster-affected people. With the emergence of online public engagement and digital volunteers, emergency responders get precise and quick intelligence which helps them to respond faster and better. Crowdsourcing makes this data available first hand to emergency decision makers ubiquitous & untampered, with the potential to contribute with situational awareness.


## What it does
The innovation aims to identify the polarity of sentiments expressed by people in social networking sites such as Twitter during and after the disaster event.

We've considered Twitter among other online social networking sites to identify such sentiments which can help emergency responders to understand and react in cases of concerns, distress, panics, information, questions etc. as well as track the change in flow of emotions from during the disaster to post-disaster relief.

## How we built it
Our intelligent disaster impact responder platform is built on Extert.ai NLP Library, QuickBase (web app) and GCP Docker Webservice. The solution is rapidly deployable and is useful to any disaster situation. For this implementation, we considered Hurricane Ian (#HurricaneIan) that hit the city of Florida on 28th September. The innovation performs a sentiment analysis of tweets posted on Twitter and covers the timeline of change in sentiments during and post hurricane.

Requirement planning was carried out by the team. We brainstormed on the fastest possible approach to communicate our idea as a product and also come up with something simple. In the end, we came up with a wireframe to guide our thoughts.

We identified the Twitter library Snscrape that helps to fetch the tweets for any given period of time. We also used Twitter API to fetch the metadata like number of likes, comments, quotes and replies. We used the expert ai

We explored and identified the different expert.ai endpoints that can be used for resource analysis and classification to derive the sentiment, behavior and emotion traits for the given text

We considered the Low-code-no-code platform Quickbase to store the data and to create analytics on the tweet data by creating a web application within the platform

We used python programming lanaguage to build the script that performs the operations like scrapping of tweet text, invoke the expert.ai endpoints through its python library and push the data to Quickbase using its RESTful JSON APIs

We have deployed the python code as a docker web service hosted in GCP cloud to execute it directly from Quickbase web application.

## Challenges we ran into

This service was our first time to know what expert.AI is and how to use it. We used lots of time to research the documents and found lots of examples to learn. When our program has some errors, we will research from documents and expert.AI NLP library to help us fix the problem.

## Accomplishments that we're proud of
We take pride in conceptualizing to developing this solution through this hackathon.

Most importantly, we take pride in the impact this product would have during disaster situations particularly the help extended towards emergency responders with precise and faster intelligence which will be of essence in time of need.

Able to build the prototype, working solution within a matter of 15 days

## What we learned
- Performing emotion traits, behavior trait and other NLP analysis
- Learnt Expert.ai NLP libraries which may help us in future as well
- How to deploy GCP docker web service making use of NLP libraries like expert.AI
- expertAI library supports only few languages but doesn't support the others like Ukranian, Japanese, Mandarin, etc.

## What's next for Disaster Resilience and Impact Responder Platform
We hope to refine the model for improved accuracy and commence efforts for further development of features with a target to provide a quality solution to the society. Also include multi-language support to cater to multiple regions.
