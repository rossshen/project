# -*- coding: utf-8 -*-
import scrapy


class SpiderSpider(scrapy.Spider):
    name = "spider"
    allowed_domains = ["http://www.thegioitraicay.net/"]
    start_urls = ['http://www.thegioitraicay.net//']

    def parse(self, response):
        pass
