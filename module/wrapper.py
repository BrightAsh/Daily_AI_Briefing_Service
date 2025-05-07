# module/wrapper.py

from module.News import News_pipeline
from module.Blogs import Blogs_pipeline
from module.Paper import Paper_pipeline

def set_web_params(n, country):
    global WEB_N, WEB_COUNTRY
    WEB_N = n
    WEB_COUNTRY = country

def News_pipeline_wrapped(keyword, days):
    return News_pipeline(keyword, days, WEB_N, WEB_COUNTRY)

def Blogs_pipeline_wrapped(keyword, days):
    return Blogs_pipeline(keyword, days, WEB_N, WEB_COUNTRY)

def Paper_pipeline_wrapped(keyword, days):
    return Paper_pipeline(keyword, days, WEB_N, WEB_COUNTRY)
