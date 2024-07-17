# Author: Eric Liao
# Date: Nov 4th, 2021

from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium.webdriver.common.by import By

##### Facility functions ########################################################
from selenium.webdriver.support.wait import WebDriverWait


def gather_information():
    jobs = driver.find_elements(By.CLASS_NAME, 'jobs-search-results__list-item')
    for job in jobs:
        driver.execute_script("arguments[0].scrollIntoView();", job)
        job.click()
        try:
            job_right_panel = driver.find_element(By.CLASS_NAME, 'jobs-search__right-rail')
        except:
            # job_right_panel = driver.find_element(By.CLASS_NAME, 'jobs-unified-top-card__content--two-pane')
            job_right_panel = driver.find_element(By.CLASS_NAME, 'jobs-details__main-content')


        # job_id0 = job_details.find_element(By.CLASS_NAME, 'jobs-unified-top-card__job-title').accessible_name
        # job_id.append(job_id0)

        try:
            job_title0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-unified-top-card__job-title').accessible_name
            job_title.append(job_title0)
        except:
            job_title.append("")

        try:
            company_name0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-unified-top-card__company-name').text
            company_name.append(company_name0)
        except:
            company_name.append("")

        try:
            location0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-unified-top-card__bullet').text
            job_location.append(location0)
        except:
            job_location.append("")

        try:
            workplace_style0 = job_right_panel.find_element(By.CLASS_NAME,
                                                            'jobs-unified-top-card__workplace-type').text
            workplace_style.append(workplace_style0)
        except:
            workplace_style.append('')

        try:
            date0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-unified-top-card__posted-date').text
            post_date.append(date0)
        except:
            post_date.append("")

        try:
            job_link0 = job_right_panel.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
            job_link.append(job_link0)
        except:
            job_link.append("")

        try:
            # job_content0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-unified-description__content').text
            job_content0 = job_right_panel.find_element(By.CLASS_NAME, 'jobs-description__content').text
            job_content.append(job_content0)
        except:
            job_content.append("")

        time.sleep(2)


def output_results(job_title, company_name, job_location, workplace_style, date, job_link, job_content):
    job_df = pd.DataFrame({"position": job_title, "company": company_name, "location": job_location,
                           "workplace_style": workplace_style, "date_posted": date, "job_link": job_link, "job_content": job_content})
    job_df.to_excel('/Users/yliao13/PycharmProjects/phd_class/web_scraping/LinkedIn_job_info_August_23rd.xlsx', index=False)
##########################################################################################################


###### Begin from here #####
driver_path = "/Users/yliao13/PycharmProjects/phd_class/web_scraping/chromedriver"

driver = webdriver.Chrome(executable_path=driver_path)

##### Linedin account info #####
username = "ericliaoyf@gmail.com"
password = "Afei99357@"

##### Job info #####
job_keywords = "recruiter, talent acquisition"
# job_keywords = "talent acquisition, campus recruiter"
# job_keywords = 'campus recruiter'
job_location = "United States"
# job_location = "Asheville"

##### specify which page you want to start to gather job info, default == 1 #####
start_page = 1

##### start processing #####
driver.get("https://www.linkedin.com/login")
time.sleep(2)

driver.find_element(By.ID, "username").send_keys(username)
driver.find_element(By.ID, "password").send_keys(password)
driver.find_element(By.ID, "password").send_keys(Keys.RETURN)

driver.get("https://www.linkedin.com/jobs/")
time.sleep(3)

##### find the keywords/location search bars #####
search_bars = driver.find_elements(By.CLASS_NAME, 'jobs-search-box__text-input')

time.sleep(3)
search_keywords = search_bars[0]
search_keywords.send_keys(job_keywords)

time.sleep(3)
search_location = search_bars[3]
search_location.send_keys(job_location)

search_button = driver.find_element(By.CLASS_NAME, 'jobs-search-box__submit-button--hidden')
driver.execute_script("arguments[0].click();", search_button)

time.sleep(5)



# # filters = driver.find_element(By.CLASS_NAME, 'search-reusables__filter-list').find_elements(By.TAG_NAME, 'li')
#
# driver.find_element(By.ID, "hoverable-outlet-date-posted-filter-value").find_element(By.CSS_SELECTOR, 'button').__setattr__('aria-expanded', 'true')
#
# # 'pl4 pr6'
# date_filters = driver.find_element(By.CLASS_NAME, 'search-reusables__colection-values-container').find_elements(By.TAG_NAME, 'li')
# date_filters[3].__setattr__('data-arttdeco-is-focused', 'true')
# driver.find_element(By.ID, 'ember1312').click()
#
# # time.sleep(3)
#
# btnAllFilterXpath = "//span[@class='artdeco-button__text' and text()='All Filters']"
# btnAllFilter = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, btnAllFilterXpath)))


##### get how many pages of results returned #####
pages = driver.find_element(By.CLASS_NAME, 'artdeco-pagination__pages').find_elements(By.TAG_NAME, 'li')
number_of_pages = len(pages)
toatal_number_pages = int(pages[number_of_pages - 1].get_attribute('data-test-pagination-page-btn'))

##### gather information for each job #####
job_title = []
company_name = []
job_location = []
workplace_style = []
post_date = []
job_link = []
job_content = []

for i in range(toatal_number_pages):
    if i < start_page - 1:
        continue

    if i < 9 and i >= start_page - 1:
        time.sleep(5)
        driver.find_elements(By.CLASS_NAME, 'artdeco-pagination__indicator')[i].find_element(By.TAG_NAME,
                                                                                             'button').click()
        time.sleep(2)
        gather_information()

        output_results(job_title, company_name, job_location, workplace_style, post_date, job_link, job_content)

    if i >= start_page - 1 and i >= 9:
        time.sleep(5)
        driver.find_elements(By.CLASS_NAME, 'artdeco-pagination__indicator')[6].find_element(By.TAG_NAME,
                                                                                             'button').click()
        time.sleep(2)
        gather_information()

        output_results(job_title, company_name, job_location, workplace_style, post_date, job_link, job_content)

output_results(job_title, company_name, job_location, workplace_style, post_date, job_link, job_content)
