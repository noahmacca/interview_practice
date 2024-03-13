# %%
import requests
from bs4 import BeautifulSoup

# crawling logic...
response = requests.get("https://scrapeme.live/shop/")
print(response)
soup = BeautifulSoup(response.content, "html.parser")
print(soup)
link_elements = soup.select("a[href]")
print(link_elements)

urls = []
for link_element in link_elements:
    url = link_element["href"]
    if "https://scrapeme.live/shop" in url:
        urls.append(url)


urls

# %%
urls = ["https://scrapeme.live/shop/"]
visited = set(urls)
products = []

while len(urls) != 0:
    current_url = urls.pop()
    print("current_url={}".format(current_url))

    response = requests.get(current_url)
    soup = BeautifulSoup(response.content, "html.parser")

    p = {
        "url": current_url,
        "image": soup.select_one(".wp-post-image")["src"],
        "title": soup.select_one(".woocommerce-loop-product__title").get_text(),
    }
    print(p)

    products.append(p)

    links = soup.select("a[href]")
    for l in links:
        url = l["href"]
        if ("https://scrapeme.live/shop" in url) and (url not in visited):
            urls.append(url)
            visited.add(url)


print(urls)
print(products)
