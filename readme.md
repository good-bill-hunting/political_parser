# :us: Good Bill Hunting

### By: Josh Holt, Adam Heywood, Bart Taylor and Jorge Lopez

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=blue"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=black"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=blue"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=black"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=blue"></a>

## :scroll: Goal: 
Aquire the text of political bills between 2001 and 2023 and use NLP to determine the political sponsor using only the body text of the bill.

## :book: Data Dictionary:
| Feature | Definition |
|:--------|:-----------|
|congressional bill| legislative proposals from the House of Representatives and Senate within the United States Congress. More info: https://www.govinfo.gov/help/bills#:~:text=A%20bill%20is%20a%20legislative,(first%20and%20second%20sessions). |
|political party| Made up of individuals who organize to win elections, operate government, and influence public policy. The Democratic and Republican parties are currently the primary parties in Congress. More info: https://www.senate.gov/reference/reference_index_subjects/Political_Parties_vrd.htm|
|sponsor| Patron, usually a legislator, who presents a bill or resolution to a legislature for consideration.|
|initial cosponsor or original cosponsor| Senator or representative who was listed as a cosponsor at the time of a bill's introduction|

## :balance_scale: How laws are made: 

https://www.house.gov/the-house-explained/the-legislative-process

Basic Steps of the legislative process:
1. First, a Representative sponsors a bill. 
2. The bill is then assigned to a committee for study. 
3. If released by the committee, the bill is put on a calendar to be voted on, debated or amended. 
4. If the bill passes by simple majority (218 of 435), the bill moves to the Senate. 
5. In the Senate, the bill is assigned to another committee and, if released, debated and voted on. 
6. If the Senate makes changes, the bill must return to the House for concurrence.  
7. The resulting bill returns to the House and Senate for final approval. 
8. The President then has 10 days to veto the final bill or sign it into law.

## :page_with_curl: Data Overview:
#### We acquired 26,000+ bills from api.govinfo.gov/.

- Target Variable: Democrat, Republican, or Independent 
- One Observation Represents: A sponsored Bill
- Initial steps: API scraping, acquiring data, creating a list of political parties

## :question: Initial Questions:
1) Are there any words unique to a specific political party in determine Congressional Bills?
2) What are the top focus areas for each political party and are there any bi-partisan areas or Congressional Bills?

## :busts_in_silhouette: To reproduce:
1. Get an api key from: https://www.govinfo.gov/api-signup
2. Append "&api_key=" to the beginning of your api key string.
3. In your env.py file, save your api key under the variable "api_key"
4. Clone the political_parser repo
5. Install the required python packages.
6. Allow several hours to acquire the data. We created another notebook, "acquire_in_chunks", to spread out the download and prevent losing the data late in the process. Ours took 10+ hours or 5ish hours running two notebooks in parallel on different chunks.
7. Once the data is acquired, run the final notebook. It takes about 20-30 minutes.

## Project Plan:
1. Acquire data from govinfo.gov's api.
2. Prepare data by dropping nulls (one row), cleaning and lemmatizing the text.
3. Explore the words each party uses.
4. Modeling using XGBoost, Decision Tree, Random Forest, KNN and Logistic Regression.
    - The accuracy will be the baseline we use for this project.


## Explore Takeaways:
- Each political party had a focus area of concentration: for democrats the main are appears to be healthcare and higher education, republicans are focused on homeland security and China, and independents tend to lean toward economic concerns.
- The large amount of bills by democrats versus the other political parties may be obscuring the data from the other parties.
- Common areas of concern appear to be health care and term limits, at least for democrats and republicans.
- Although specific words may not necessarily determine if a bill is a certain political party, there are some words that are associated with particular political interests that could determine if a bill is from a particular political party.

## Modeling Takeaways:
- Our baseline was predicting Democrat at 59%
- Four of the five models performed better than the baseline on train and validate.
- The Logistic Regression model performed the best on train (75%) and validate (71%).
- We chose to move forward with the Logistic Regression model because KNN may be overfit.

## Recommendations:
- The results from this project should be made public for informational use on the focus of democratic and republican representatives.
- An investigation or inquiry should be established to determine why there is an imbalance of bills proposed between the two major political parties.

## :footprints: Next Steps:
- This project can be used as a starting point for a larger project on time series where a team can add other NLP bills throughout history and future bills to analyze the change in political areas of concentration over time.
- Investigate which bills have made it through the process and are now laws and build a model that would predict, based off of the language, whether a bill is made into law or not.
- Remove the dates from the bills and make another column to sort by the dates in order to determine if there is any change or difference in language between certain timeframes or major events.