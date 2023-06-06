import requests
import re
import os
import csv
import time
from bs4 import BeautifulSoup

AnimeNum = 0

def CrawlInfo(url,name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")
    if AnimeNum < 14: 
        score = soup.find('div',class_ = 'score-label score-9').text
    else:
        score = soup.find('div',class_ = 'score-label score-8').text

    tmp = soup.find_all('span',{'itemprop':'genre'})
    tags = []
    for i in tmp:
        tags.append(i.text)

    with open('anime.csv','a+',encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name,tags,score])
        writer = None


response = []
soup = []

response.append(requests.get('https://myanimelist.net/topanime.php'))
soup.append(BeautifulSoup(response[0].text,"html.parser"))

# Top 200 Anime List
for i in range(1,4):
    num = i*50
    response.append(requests.get('https://myanimelist.net/topanime.php' + '?limit=' + str(num)))
    soup.append(BeautifulSoup(response[i].text,"html.parser"))

with open('Top200.html','w',encoding='utf-8') as file:
    file.write(str(soup[0])+str(soup[1])+str(soup[2])+str(soup[3]))

# Get the urls of Top200 from the top50 html
urls = []

with open('Top200_url.txt','w',encoding='UTF-8') as file:
    for i in range (0,4):
        animes = soup[i].select("h3 a")
        count = 0
        for sub in animes:
            if count < 50:
                file.write(sub['href']+'\n')
                urls.append(sub['href'])
                count += 1

with open('anime.csv',"w",encoding='UTF-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Anime', 'Genres','Rating'])
        writer= None

for link in range(117,200):
    count = 0
    name = ''   
    for c in urls[link]:
        if c == '/' and count < 5:
            count += 1
        elif count >= 5:
            name += c
    CrawlInfo( urls[link], name )
    AnimeNum += 1
    print('Number Of Anime Is Done : ' + str(link + 1))
    if(AnimeNum % 100 == 0):
        time.sleep(60)


#CrawlInfo('https://myanimelist.net/anime/34096/Gintama','Gintama')