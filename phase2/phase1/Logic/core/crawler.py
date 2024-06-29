import requests 
import bs4

from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re 

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': None
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = set()
        self.crawled_data = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()


    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        if URL[0:29] == "https://www.imdb.com/title/tt" : 
            return URL.split('/')[4]
        else : 
            return None
    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        with open('IMDB_crawled.json','w') as f : 
            json.dump(list(self.crawled_data), f)
        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(list(self.not_crawled), f)
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled_data = json.load(f)

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = deque(json.load(f))

        self.added_ids = set()
        for movie in self.crawled_data : 
            movie_id = movie['id'] 
            self.added_ids.add(movie_id)
            self.crawled.add('https://www.imdb.com/title/' + movie_id + '/')

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # TODO
        # it may need header 
       # print("=-----=")
      #  print("---",URL)
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
        response = requests.get(URL,headers=headers)
     #   if response.status_code == 308 : 
     #       new_url = response.headers['Location']
      #      print(new_url)
       #     response = requests.get(new_url,headers=headers)
      #  print(response.text)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL) 
      #  print(response.status_code)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            #link_elements = 
            movies = soup.select("a[href]")
           # print(movies)
            for movie in movies:
              #  print("----------")
            #    print(movie)
                s = movie['href']
                f = s.split('/')
           #     print(f)
                if f[1] == 'title' and f[2][0:2] == 'tt' :  
                     movie_id = f[2] 
                else : 
                    continue

           #     print("---")
            #    print(movie["href"])
             #   print("***", movie_id)
                if movie_id not in self.added_ids:
                    self.added_ids.add(movie_id)
                    self.not_crawled.append('https://www.imdb.com/title/' + movie_id + '/')
        print(len(self.added_ids))
       # print(self.not_crawled)
        print('the end')

    def get_imdb_instance(self,soup,url):
      #  print('chera?')
     #   print('url:',url)
       # print('soup:',soup)
   #     print("harsh : --", self.get_summary(soup))
    #    print('bad?')
        
        imdb_instance = {
            'id': self.get_id_from_URL(url),
            'title': self.get_title(soup),
            'first_page_summary': self.get_first_page_summary(soup),
            'release_year': self.get_release_year(soup),
            'mpaa': self.get_mpaa(soup),
            'budget': self.get_budget(soup),
            'gross_worldwide': self.get_gross_worldwide(soup),
            'rating': self.get_rating(soup),
            'directors': self.get_director(soup),
            'writers': self.get_writers(soup),
            'stars': self.get_stars(soup),
            'related_links': self.get_related_links(soup),
            'genres': self.get_genres(soup),
            'languages': self.get_languages(soup),
            'countries_of_origin': self.get_countries_of_origin(soup),
            'summaries': self.get_summary(url),
            'synopsis': self.get_synopsis(url),
            'reviews': self.get_reviews_with_scores(url)
        }
       # print('choon ch chasbide be ra')
        return imdb_instance

    def extract_more_movie(self) : 
        self.added_ids.add('tt1924429')
        self.not_crawled.append('https://www.imdb.com/title/tt1924429/')
        self.added_ids.add('tt0002011')
        self.not_crawled.append('https://www.imdb.com/title/tt0002011/')
        self.added_ids.add('tt0414853')
        self.not_crawled.append('https://www.imdb.com/title/tt0414853/')

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        
        # help variables
        NEW_URL = None
        THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        self.extract_more_movie()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while len(self.crawled_data) < self.crawling_threshold and len(self.not_crawled) > 0:
                 
                new_url = self.not_crawled.popleft()
         #       print('num crawled', len(self.crawled))
        #        print('num not_crawled', len(self.not_crawled))
                                
                if new_url not in self.crawled:
                  #  response = self.crawl(new_url)
                 #   print(new_url, response)
                 #   if response.status_code == 200:
                     #   print(new_url)
                        # Parse the response and extract information
                        # Update crawled and not_crawled lists accordingly
                    futures.append(executor.submit(self.crawl_page_info, new_url))
                  #  self.crawled.add(new_url)
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []
     #   self.write_to_file_as_json()

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        if len(self.crawled_data) > self.crawling_threshold : 
            return
        
        print("new iteration")
        print('num crawled', len(self.crawled_data))
       # print(URL)
        response = self.crawl(URL)
      #  print(URL,response)
        if response.status_code == 200:
        #    print("-4-4,",URL)
            soup = BeautifulSoup(response.text, 'html.parser')
         #   print('soup')
          #  print(self.get_imdb_instance(soup,URL))
            movie = self.get_imdb_instance(soup,URL)
          #  print('kore khar')
         #   movie['id'] = self.get_id_from_URL(URL)
            #print('khar')
            movie = self.extract_movie_info(response, movie, URL)
            #print('=====================')
              #  print('*************************')
            
            self.crawled.add(URL)
            do = 1
            for field in movie.keys() : 
                if movie[field] == None :
                    do = 0 
            if do :
                self.crawled_data.append(movie) 

       #     print('-----------------')
     #       print(movie) 
        #    exit(0)
         #   print('-----------------')

            related_links = self.get_related_links(soup)
            # print('---',related_links)
            for link in related_links:
                #   print(link)
                movie_id = self.get_id_from_URL(link)
                #  print(movie_id)
              #  print('***',link)
                if movie_id not in self.added_ids and link not in self.crawled:
               #     print('---', link)
                #    print('mirese')
                    self.not_crawled.append(link)
                 #   print("-------------")
                  #  print('khare', len(self.not_crawled))
                    self.added_ids.add(movie_id)
    pass

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        soup = BeautifulSoup(res.text, 'html.parser')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['rating'] = self.get_rating(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['summaries'] = self.get_summary(URL)
        movie['synopsis'] = self.get_synopsis(URL)
        movie['reviews'] = self.get_reviews_with_scores(URL)

        return movie

    def get_summary_link(self,url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return "https://www.imdb.com/title/" + self.get_id_from_URL(url) + '/plotsummary/'
        except:
            print("failed to get summary link")

    def get_review_link(self,url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return "https://www.imdb.com/title/" + self.get_id_from_URL(url) + '/reviews/'
        except:
            print("failed to get review link")

    def get_title(self,soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
  #      try:
       # print('are')
        title_tag = soup.find('title') 
        #print('not')
        #  print('aerg',title_tag)
        title = title_tag.text if title_tag else None
        #print(title)
        return title 
   # except:
    #        print("failed to get title")

    def get_first_page_summary(self,soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
          #  print(script_tag)
            if script_tag:
            #    print("---")
                data = json.loads(script_tag[0].string)
             #   print("***",data)
                return data['description']
            '''
            summary_tag = soup.find_all('meta',{'name':'description'})
            summary = summary_tag[0]['content'] if summary_tag else None
            return summary.strip() if summary else None
            '''
        except:
            print("failed to get first page summary")

    def get_director(self,soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            script_tag = soup.find('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag.string)
                directors = [director['name'] for director in data.get('director', [])]
            return directors
        except:
            print("failed to get director")

    def get_stars(self,soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
      #      print(script_tag)
            if script_tag:
              #  print("---")
                data = json.loads(script_tag[0].string)
              #  print("***",data)
                stars = [actor['name'] for actor in data.get('actor', [])]
                return stars
        except:
            print("failed to get stars")

    def get_writers(self,soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})

            data = json.loads(script_tag.string)
            data = data['props']['pageProps']['mainColumnData']['writers']
            writers = [] 
            for tag in data : 
                tag = tag['credits'] 
                for name_tag in tag : 
                    wrtier = name_tag['name']['nameText']['text']
                    writers.append(wrtier)
            #print(data)
            return writers
        except:
            print("failed to get writers")

    def get_related_links(self,soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            urls = []
            link_elements = soup.select("a[href]")
            for link_element in link_elements:
                url = link_element['href']
             #   print(url)
            # print(url)
                if "/title/tt" == url[0:9] and (len(url.split('/')) == 3 or len(url.split('/')) == 4):
                    if 'https://' not in url : 
                        url = 'https://www.imdb.com' + url
                #    print(url)
                    urls.append(url + '/')
           # print(urls)
            return urls
        except:
            print("failed to get related links")

    def get_summary(self,url):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            url = self.get_summary_link(url) 
            response = self.crawl(url)      
            soup = BeautifulSoup(response.text, 'html.parser')
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            data = json.loads(script_tag.string)
            categories = data['props']['pageProps']['contentData'].get('categories', [])
            summaries = []
            for category in categories:
                if category.get('id') == 'summaries':
                    section = category.get('section', {})
                    items = section.get('items', [])
                    for item in items:
                        html_content = item.get('htmlContent', '')
                        summary = BeautifulSoup(html_content, 'html.parser').get_text(strip=True)
                        summaries.append(summary)
            return summaries
        except:
            print("failed to get summary")

    def get_synopsis(self,url):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            url = self.get_summary_link(url) 
            response = self.crawl(url)      
            soup = BeautifulSoup(response.text, 'html.parser')
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            data = json.loads(script_tag.string)
           # print(data)
            categories = data['props']['pageProps']['contentData'].get('categories', [])
            summaries = []
           # print(categories)
            for category in categories:
                if category.get('id') == 'synopsis':
                    section = category.get('section', {})
                    items = section.get('items', [])
                    for item in items:
                        html_content = item.get('htmlContent', '')
                        summary = BeautifulSoup(html_content, 'html.parser').get_text(strip=True)
                        summaries.append(summary)
            return summaries
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self,url):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            url = self.get_review_link(url)
            response = self.crawl(url) 
            soup = BeautifulSoup(response.text,'html.parser')
            reviews = []
            review_containers = soup.find_all('div', class_='lister-item-content')
            for container in review_containers:
                content_text = None
               
                content = container.find('div', class_='text show-more__control')
                if content:
                    content_text = content.text.strip()
                else:
                    content_text = "Content not found"

                score_number = None
                score_container = container.find('span', class_='rating-other-user-rating')
                if score_container:
                    score = score_container.find('span', class_='point-scale')
                    if score:
                        score_number = score.text.strip()
                    else:
                        score_number = "Score not found"
                else:
                    score_number = "Score not found"

                reviews.append([content_text,score_number])

            return reviews

        except:
            print("failed to get reviews")

    def get_genres(self,soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag[0].string)
                return data['genre']
        except:
            print("Failed to get generes")

    def get_rating(self,soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag[0].string)
                return str(data['aggregateRating']['ratingValue'])
        except:
            print("failed to get rating")

    def get_mpaa(self,soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag[0].string)
                return data['contentRating']
        except:
            print("Error occurred while getting MPAA rating:")
        

    def get_release_year(self,soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            script_tag = soup.find_all('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag[0].string)
                return data['datePublished']
        except:
            print("failed to get release year")

    def get_languages(self,soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            languages = []
            language_tags = soup.find_all('a', href=re.compile(r'/search/title\?title_type=feature&primary_language=.*'))
            for tag in language_tags:
                languages.append(tag.get_text(strip=True))
            return languages

        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self,soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            countries = []
            origin_span = soup.find('span', text='Countries of origin')
            if origin_span:
                country_tags = origin_span.find_next('ul', class_='ipc-inline-list').find_all('a')
                for tag in country_tags:
                    country = tag.get_text(strip=True)
                    if country:
                        countries.append(country)
            return countries if countries else None


        except:
            print("failed to get countries of origin")

    def get_budget(self,soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget_tag = soup.find('span', class_='ipc-metadata-list-item__list-content-item', text=re.compile(r'^\$.*'))
            budget = budget_tag.text.strip() if budget_tag else None
            return budget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self,soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross_tag = soup.find('li', class_='ipc-metadata-list__item', attrs={'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
            gross = gross_tag.find('span', class_='ipc-metadata-list-item__list-content-item').text.strip() if gross_tag else None
            return gross
        except:
            print("failed to get gross worldwide")


def main():
    
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    imdb_crawler.read_from_file_as_json()
    
  #  for ids in imdb_crawler.added_ids : 
   #     if ids == 'tt0137523' : 
    #        print(ids)
   # url = 'https://www.imdb.com/title/tt0109830/?ref_=tt_sims_tt_t_2'
    #response = imdb_crawler.crawl(url)
    #soup = BeautifulSoup(response.text, 'html.parser')
    #script_tag = soup.find_all('script', type='application/ld+json')
    #data = json.loads(script_tag[0].string)
    #mpaa_rating_element = soup.find('div', class_='subtext')

    #print(data)
  #  print(response.text)
    #related_links = imdb_crawler.get_related_links(soup) 
    #print(imdb_crawler.get_mpaa(soup))
    #print("------------")
    
    imdb_crawler.start_crawling()
    print('pending...')
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
