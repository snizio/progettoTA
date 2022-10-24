import re
from collections import namedtuple
import requests
from bs4 import BeautifulSoup

UserReview = namedtuple('UserReview', ['product_name', 'review_title', 'comment', 'rating', 'date', 'username', 'profile_url', 'verified_purchase'])

class AmazonScraper:
    review_date_pattern = re.compile('(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d+, \d{4}')
    product_name_pattern = re.compile('^https:\/{2}www.amazon.com\/(.+)\/product-reviews')
    def __init__(self):
        # create a browser session
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv: 87.0) Gecko/20100101 Firefox/87.0'
        self.session.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        self.session.headers['Accept-Language'] = 'en-US,en;q=0.5'
        self.session.headers['Connection'] = 'keep-alive'
        self.session.headers['Upgrade-Insecure-Requests'] = '1'

    def scrapeReviews(self, url, page_num, filter_by='recent'):
        """
        args
            filter_by: recent or helpful
        return
            namedtuple
        """
        try:
            review_url = re.search('^.+(?=\/)', url).group()
            review_url = review_url + '?reviewerType=all_reviews&sortBy={0}&pageNumber={1}'.format(filter_by, page_num)
            print('Processing {0}...'.format(review_url))
            response = self.session.get(review_url)

            product_name = self.product_name_pattern.search(url).group(1) if self.product_name_pattern.search(url) else ''
            if not product_name:
                print('url is invalid. Please check the url.')
                return
            else:
                product_name = product_name.replace('-', ' ')

            soup = BeautifulSoup(response.content, 'html.parser')
            review_list = soup.find('div', {'id': 'cm_cr-review_list' })    

            reviews = []
            product_reviews = review_list.find_all('div', {'data-hook': 'review'}) # return reviews
            for product_review in product_reviews:
                review_title = product_review.find('a', {'data-hook': 'review-title'}).text.strip()
                verified_purchase = True if product_review.find('span', {'data-hook': 'avp-badge'}) else False
                review_body = product_review.find('span', {'data-hook': 'review-body'}).text.strip()
                rating = product_review.find('i', {'data-hook': 'review-star-rating'}).text
                review_date = self.review_date_pattern.search(product_review.find('span', {'data-hook': 'review-date'}).text).group(0)
                username = product_review.a.span.text
                user_profile = 'https://amazon.com/{0}'.format(product_review.a['href'])
                reviews.append(UserReview(product_name, review_title, review_body, rating, review_date, username, user_profile, verified_purchase))
            return reviews
        except Exception as e:
            print(e)
            return None