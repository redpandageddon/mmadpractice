from bs4 import BeautifulSoup as bs
import requests
import numpy as np

pages = []

for x in range(1, 15):
    try:
        pages.append(requests.get('https://stopgame.ru/review/new/izumitelno/p' + str(x)))
    except (Exception):
        break

comments = []
articles = 0

for r in pages:
    html = bs(r.content, 'html.parser')
    
    for el in html.select('.item.article-summary'):
        title = el.select('.caption > a')
        print('[Статья] ' + title[0].text)
        articles += 1
        comment = el.select('.info-item.comments a')
        
        try:
            comments.append(comment[0].text)
            print('Количество комментариев ' + comment[0].text)
        except (Exception):
            continue
        
arr = np.array(comments).astype(np.float)

print('[Данные] Общее количество статей ' + str(articles) + '\n')
print('[Данные] Среднее количество комментариев ' + str(np.mean(arr))  + '\n')
print('[Данные] Максмиальное количество комментариев ' + str(np.max(arr))  + '\n')
print('[Данные] Общее количество комментариев ' + str(sum(arr))  + '\n')