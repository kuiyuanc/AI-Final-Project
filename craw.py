import requests
import re
import os
import csv
from bs4 import BeautifulSoup

def scrawOnePage(url,name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")
    # with open( name + ".html","w", encoding='UTF-8') as file:
    #     file.write(str(soup))

    pattern = re.compile(r'<[^>]+>',re.S)
    reviews = str(soup.select(".text"))
    #get rating
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
    time = 0
    tmp = ''
    flag = 0
    for s in result.split('$#$'):
        if flag == 0:
            flag = 1
        else:
            table.append([name,s,rating_list[time]])
            tmp += s
            time += 1

    with open('review.csv',"a+",encoding='UTF-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)
    
    # with open( name + "_review.txt","w", encoding='UTF-8') as file:
    #     file.write(tmp)

#open top50 anime page and get the html
response = requests.get('https://myanimelist.net/topanime.php')
soup = BeautifulSoup(response.text,"html.parser")
# with open('top50.html',"w",encoding='UTF-8') as file:
#     file.write(str(soup))

#get the urls of top50 from the top50 html
animes = soup.select("h3 a")
urls = []
count = 0
with open('reviews/top50_url.txt','w',encoding='UTF-8') as file:
    for sub in animes:
        if count < 50:
            file.write(sub['href']+'\n')
            urls.append(sub['href'])
            count += 1
#get reviews of top50 animes
with open('review.csv',"a+",encoding='UTF-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Anime', 'Reviews','Rating'])
for url in urls:
    count = 0
    name = ''   
    for c in url:
        if c == '/' and count < 5:
            count += 1
        elif count >= 5:
            name += c
    scrawOnePage(url + '/reviews',name)

#scrawOnePage('https://myanimelist.net/anime/9253/Steins_Gate/reviews','Steins_Gate')
print("Done")