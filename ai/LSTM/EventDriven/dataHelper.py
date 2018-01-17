import requests
import sys, csv, json
import numpy as np
import pandas as pd
from datetime import datetime

reload(sys)
sys.setdefaultencoding('utf8')

class APIKeyException(Exception):
    def __init__(self, message):
        self.message = message

class InvalidQueryException(Exception):
    def __init__(self, message):
        self.message = message

class NoAPIKeyException(Exception):
    def __init__(self, message):
        self.message = message

class ArchiveAPI(object):
    def __init__(self, key=None):
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None, ):
        """
        Calls the archive API and returns the results as a dictionary.
        :param key: Defaults to the API key used to initialize the ArchiveAPI class.
        """
        if not key:
            key = self.key

        if (year < 1882) or not (0 < month < 13):
            # currently the Archive API only supports year >= 1882
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)

        url = self.root.format(year, month, key)
        requests.adapters.DEFAULT_RETRIES = 20
        r = requests.get(url)
        return r.json()


api = ArchiveAPI('0ba6dc04a8cb44e0a890c00df88c393a')

years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dataDirPath = '/home/EventDrivenLSTM/data/'

def queryNewsFromNYTimes():
    for year in years:
        for month in months:
            print('year', year, 'month', month)
            mydict = api.query(year, month)
            print('mydict', mydict)
            file_str = dataDirPath + str(year) + '-' + '{:02}'.format(month) + '.json'
            with open(file_str, 'w') as fout:
                try:
                    json.dump(mydict, fout)
                except:
                    pass
            fout.close()

# ################# Query Stock price #################

def interpolateDf(df):
    df1 = df
    idx = pd.date_range('12-29-2006', '12-31-2016')
    df1.index = pd.DatatimeIndex(df1.index)
    df1 = df1.reindex(idx, fill_value=np.NaN)
    interpolated_df = df1.interpolate()
    interpolated_df.count()
    return interpolated_df

def queryStockPrice():
    with open(
            '/Users/georgezou/Documents/Coding/ML/Neural-Network-with-Financial-Time-Series-Data/data/DJIA_indices_data.csv',
            'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # Converting the csv file reader to a lists
        data_list = list(spamreader)

    # Seoarating header from the data
    header = data_list[0]
    data_list = data_list[1:]
    data_list = np.asarray(data_list)

    print(data_list)

    # Selecting date and close value and adj close for each day
    # Volume will be added in the next model
    selected_data = data_list[:, [0, 4, 6]]

    # Convert it into dataframe
    # index = date
    df = pd.DataFrame(data=selected_data[0:, 1:],
                      index=selected_data[0:, 0],
                      columns=['close', 'adj close'],
                      dtype='float64')

    return interpolateDf(df)

# ########### Merging Data ################
date_format = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+%f"]

years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dict_keys = ['pub_date', 'headline'] #, 'lead_paragraph']
articles_dict = dict.fromkeys(dict_keys)

# Filtering to read only the following news

type_of_material_list = ['blog', 'brief', 'news', 'editorial', 'op-ed', 'list','analysis']
section_name_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health']
news_desk_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health', 'foreign']

current_date = '2016-10-01'
current_article_str = ''

def try_parsing_date(text):
    for fmt in date_format:
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    raise ValueError('no valid date format found')

def saveToFile(interpolated_df):
    # Saving the data as pickle file
    interpolated_df.to_pickle('/Users/georgezou/Desktop/stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl')

    # Save pandas frame in csv form
    interpolated_df.to_csv(
        '/Users/georgezou/Desktop/stock_rnn_data/sample_interpolated_df_10_years_filtered_lead_para.csv',
        sep='\t', encoding='utf-8')

    # Reading the data as pickle file
    dataframe_read = pd.read_pickle('/Users/georgezou/Desktop/stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl')
    return dataframe_read

'''
Search for every month
'''
def mergingData(interpolated_df):
    # Adding article column to dataframe
    interpolated_df["articles"] = ''

    count_articles_filtered = 0
    count_total_articles = 0
    count_main_not_exist = 0
    count_unicode_error = 0
    count_attribute_error = 0

    for year in years:  # search for every month
        for month in months:
            file_str = '/Users/georgezou/Desktop/stock_rnn_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'

            with open(file_str) as data_file:
                NYTimes_data = json.load(data_file)

            count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])  # add article number
            for i in range(len(
                    NYTimes_data["response"]["docs"][:])):  # search in every docs for type of material or section = in the list
                try:
                    if any(substring in NYTimes_data["response"]["docs"][:][i]['type_of_material'].lower() for substring in
                           type_of_material_list):
                        if any(substring in NYTimes_data["response"]["docs"][:][i]['section_name'].lower() for substring in
                               section_name_list):
                            # count += 1
                            count_articles_filtered += 1
                            # print 'i: ' + str(i) dick_key = ['pub_date', 'headline']
                            articles_dict = {your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in
                                             dict_keys}
                            articles_dict['headline'] = articles_dict['headline']['main']  # Selecting just 'main' from headline
                            # articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                            date = try_parsing_date(articles_dict['pub_date'])
                            # print 'article_dict: ' + articles_dict['headline']
                            # putting same day article str into one str
                            if date == current_date:
                                current_article_str = current_article_str + '. ' + articles_dict['headline']
                            else:
                                interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[
                                    current_date, 'articles'] + '. ' + current_article_str)
                                current_date = date
                                # interpolated_df.set_value(date, 'articles', current_article_str)
                                # print str(date) + current_article_str
                                current_article_str = articles_dict['headline']
                            # For last condition in a year
                            if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1):
                                interpolated_df.set_value(date, 'articles', current_article_str)

                                # Exception for section_name or type_of_material absent
                except AttributeError:
                    # print 'attribute error'
                    # print NYTimes_data["response"]["docs"][:][i]
                    count_attribute_error += 1
                    # If article matches news_desk_list if none section_name found
                    try:
                        if any(substring in NYTimes_data["response"]["docs"][:][i]['news_desk'].lower() for substring in
                               news_desk_list):
                            # count += 1
                            count_articles_filtered += 1
                            # print 'i: ' + str(i)
                            articles_dict = {your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in
                                             dict_keys}
                            articles_dict['headline'] = articles_dict['headline']['main']  # Selecting just 'main' from headline
                            # articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                            date = try_parsing_date(articles_dict['pub_date'])
                            # print 'article_dict: ' + articles_dict['headline']
                            if date == current_date:
                                current_article_str = current_article_str + '. ' + articles_dict['headline']
                            else:
                                interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[
                                    current_date, 'articles'] + '. ' + current_article_str)
                                current_date = date
                                # interpolated_df.set_value(date, 'articles', current_article_str)
                                # print str(date) + current_article_str
                                current_article_str = articles_dict['headline']
                            # For last condition in a year
                            if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1):
                                interpolated_df.set_value(date, 'articles', current_article_str)

                    except AttributeError:
                        pass
                    pass
                except KeyError:
                    print ('key error')
                    # print NYTimes_data["response"]["docs"][:][i]
                    count_main_not_exist += 1
                    pass
                except TypeError:
                    print ("type error")
                    # print NYTimes_data["response"]["docs"][:][i]
                    count_main_not_exist += 1
                    pass

    print (count_articles_filtered)
    print (count_total_articles)
    print (count_main_not_exist)
    print (count_unicode_error)

    # Putting all articles if no section_name or news_desk not found
    for date, row in interpolated_df.T.iteritems():
        if len(interpolated_df.loc[date, 'articles']) <= 400:
            # print interpolated_df.loc[date, 'articles']
            # print date
            month = date.month
            year = date.year
            file_str = '/Users/georgezou/Desktop/stock_rnn_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
            with open(file_str) as data_file:
                NYTimes_data = json.load(data_file)
            count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
            interpolated_df.set_value(date.strftime('%Y-%m-%d'), 'articles', '')
            for i in range(len(NYTimes_data["response"]["docs"][:])):
                try:

                    articles_dict = {your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in
                                     dict_keys}
                    articles_dict['headline'] = articles_dict['headline']['main']  # Selecting just 'main' from headline
                    # articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                    pub_date = try_parsing_date(articles_dict['pub_date'])
                    # print 'article_dict: ' + articles_dict['headline']
                    if date.strftime('%Y-%m-%d') == pub_date:
                        interpolated_df.set_value(pub_date, 'articles',
                                                  interpolated_df.loc[pub_date, 'articles'] + '. ' + articles_dict[
                                                      'headline'])

                except KeyError:
                    print ('key error')
                    # print NYTimes_data["response"]["docs"][:][i]
                    # count_main_not_exist += 1
                    pass
                except TypeError:
                    print ("type error")
                    # print NYTimes_data["response"]["docs"][:][i]
                    # count_main_not_exist += 1
                    pass

    saveToFile(interpolated_df)

    return count_articles_filtered, count_total_articles, count_main_not_exist, count_unicode_error

