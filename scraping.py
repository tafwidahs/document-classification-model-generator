from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

def get_soup(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    return 

#Moving to another page
def get_url_next_page(soup):
    """
        Checks the current active page and returns the URL of the next page.
        If it is the last page, then return None.
    """
    
    article_pager = soup.find('div', class_='article_pager')
    if article_pager is not None:
        hyperlink_tags = article_pager.find_all('a')
        for i, hyperlink_tag in enumerate(hyperlink_tags):
            if 'active' in hyperlink_tag['class']:
                try:
                    url_next_page = hyperlink_tags[i+1]['href']
                    return url_next_page
                except IndexError:
                    print('Last page reached')
    return None

#Scrape Date, Title, Category, and hyperlink for each document
def scrape_judgments(soup, categories):
    """
        Scrape the date, title and hyperlink of the individual judgment from a webpage.
    """
    
    judgments = []
    article_tags = soup.find_all('article', class_='edn_clearFix edn_simpleList')
    for article_tag in article_tags:
        found = 0
        metadata = article_tag.find('div').text
        for cat in categories:
            if cat in metadata:
                category = cat
                found = 1
                break
        if found == 1:
            hyperlink_tag = article_tag.find('h4').find('a')
            title = hyperlink_tag.text
            url = hyperlink_tag['href'].replace(' ', '%20')
            date = metadata.split('Decision Date:')[1] if len(metadata.split('Decision Date:')) > 1 else ''
            article_info = {'date': date.rstrip(), 'title': title.replace('\n', ''), 'url': url, 'category': category}
            judgments.append(article_info)
        
    return judgments

def run(url, categories):
    judgment_list = []
    soup = get_soup(url)
    judgments = scrape_judgments(soup, categories)
    judgment_list.extend(judgments)

    while True:
        url_next_page = get_url_next_page(soup)
        if url_next_page is None:
            break

#         print('Scraping page {}'.format(url_next_page.split('/')[-1].split('?')[0]))
        soup = get_soup(url_next_page)
        judgments = scrape_judgments(soup, categories)
        judgment_list.extend(judgments)
    print(len(judgment_list))

    return judgment_list

# URLs for first page of the various judgment categories
url_coa = 'https://www.singaporelawwatch.sg/Judgments/Court-of-Appeal'

print('Extracting judgments for Court of Appeal')
category = ['Insolvency', 'Family', 'Criminal']
judgment_list_coa = run(url_coa, category)
print(judgment_list_coa[0])

def download_pdf(judgment_list, folder_name):
    for judgment in judgment_list:
        response = requests.get(judgment['url'])
        folderpath = './{}_Judgements/{}'
        filepath = './{}_Judgements/{}/{}'
        filename = judgment['url'].split('/')[-1].replace('%20',' ')
        #filename = ''.join(x for x in judgment['title'] if x.isalnum() or x in '.,_-[] ')
        print(filename)
        if not os.path.isdir(folderpath.format(folder_name, judgment['category'])):
            os.makedirs(folderpath.format(folder_name, judgment['category']))
        if (len(filename) > 111):
            shortened_filename = '[' + filename.split('[')[-1]
            filename = shortened_filename
        with open(filepath.format(folder_name, judgment['category'], filename), 'wb') as f:
            f.write(response.content)

download_pdf(judgment_list_coa, 'Court_of_Appeal')