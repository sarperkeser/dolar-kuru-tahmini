from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv
import pandas as pd

def veri_cek():
    
    sayfa = int(input("scroll sayısını girin = "))

    driver_path = "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
    browser = webdriver.Chrome(driver_path)

    browser.get("https://twitter.com/search?q=ekonomi%20(from%3AProfDemirtas)%20-filter%3Alinks%20-filter%3Areplies&src=typed_query")
    
    #
    file = open("tweetler5.csv","w",encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["tweetler","tarih"])
    
    
    #
    a = 0
    while a < sayfa:
    #
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        i=0
        while i<1:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            newHeight = browser.execute_script("return document.body.scrollHeight")

            if newHeight == lastHeight:
                break
            else:
                lastHeight = newHeight

            i = i+1
    #



        sayfa_kaynağı = browser.page_source
        soup = BeautifulSoup(sayfa_kaynağı, "html.parser")
        tweetler = soup.find_all("div",attrs={"data-testid":"tweet"})


        for i in tweetler:
            
            try:
                yazı = i.find("div", attrs={"class" : "css-901oao r-18jsvk2 r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0"}).text
                tarih1 =i.find("div", attrs={"class":"css-1dbjc4n r-1d09ksm r-18u37iz r-1wbh5a2"})
                tarih=tarih1.find("a", attrs={"class":"css-4rbku5 css-18t94o4 css-901oao r-m0bqgq r-1loqt21 r-1q142lx r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-3s2u2q r-qvutc0"}).text
                writer.writerow([yazı,tarih])
                
            
            except:
                print("**")
        a = a+1
        

veri_cek()

ss = pd.read_csv("tweetler5.csv")

ss.to_excel("tweetler_excel5.xlsx")
