import json
import requests
import sys, csv, json

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

