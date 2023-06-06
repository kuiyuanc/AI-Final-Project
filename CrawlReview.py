import requests
import re
import os
import csv
import time
from bs4 import BeautifulSoup

Timer = 1
ReviewNum = 0
AnimeNum = 0
def CrawlReview(url,name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")
    # with open( name + ".html","w", encoding='UTF-8') as file:
    #     file.write(str(soup))

    pattern = re.compile(r'<[^>]+>',re.S)
    reviews = str(soup.select(".text"))
    # Get rating
    rating = soup.find_all('div',class_ = "rating mt20 mb20 js-hidden")
    rating_list = []
    for num in rating:
        rating_list.append(num.text)
    result = ''

    for substr in reviews.split('\n'):
        if '<div class="text">' in substr:
            index = substr.find('<div class="text">') + 18
            result += (substr[:index] + '$#$' + substr[index:])
        else:
            result += substr + ' '
        
    result = pattern.sub('' , result)
    table = []
    times = 0
    flag = 0
    global ReviewNum
    for s in result.split('$#$'):
        if flag == 0:
            flag = 1
        else:
            table.append([name,s,rating_list[times]])
            ReviewNum += 1
            times += 1

    with open('review.csv',"a+",encoding='UTF-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)
        writer= None
    
    global Timer 
    Tool = soup.find_all('div', class_ = 'ml4 mb8' )
    flag = 0
    NextUrl = ''
    l = Tool[1].find_all('a')
    for t in l:
        if t.text == 'More Reviews' and flag == 0:
            NextUrl = t['href']
            flag = 1
    
    if(flag == 1 and NextUrl != ''):
        Timer += 1 
        if Timer % 40 == 0:
            time.sleep(60)
            print('Little Sleep Done')
        CrawlReview(NextUrl,name)
        
        



# Open Top50 anime page and get the html

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


# Get reviews of Top200 animes
with open('review.csv',"w",encoding='UTF-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Anime', 'Reviews','Rating'])
        writer= None


# Call crawl function

for link in range(0,200):
    count = 0
    name = ''   
    for c in urls[link]:
        if c == '/' and count < 5:
            count += 1
        elif count >= 5:
            name += c
    CrawlReview( urls[link] + '/reviews', name )
    print(name + ' Done' + ' Reviews:' + str(ReviewNum))
    ReviewNum = 0
    AnimeNum += 1
    if(AnimeNum % 10 == 0):
        time.sleep(120)
    else:
        time.sleep(60)
    
    print('Big Sleep Done')
    print('Number Of Anime Is Done : ' + str(link + 1))

#CrawlReview('https://myanimelist.net/anime/41467/Bleach__Sennen_Kessen-hen/reviews','Bleach__Sennen_Kessen-hen')
print("Done")

