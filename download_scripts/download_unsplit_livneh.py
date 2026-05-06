import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup

def download_files_within_year_range(url, output_dir, start_year, end_year, file_extensions=None):
    """
    Downloads files from a webpage within a specific year range using wget.

    :param url: URL of the webpage to scrape.
    :param output_dir: Directory to save the downloaded files.
    :param start_year: Start year for filtering files.
    :param end_year: End year for filtering files.
    :param file_extensions: List of file extensions to filter (e.g., ['.nc']).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the HTML content of the webpage
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links on the page
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # Filter links by file extension if specified
    if file_extensions:
        pattern = re.compile(rf"({'|'.join(re.escape(ext) for ext in file_extensions)})$")
        links = [link for link in links if pattern.search(link)]

    # General regex to capture any four-digit year
    year_pattern = re.compile(r'(19\d{2}|20\d{2}|21\d{2})(?!.*(19\d{2}|20\d{2}|21\d{2}))')


    # Filter links by year range, defaulting to downloading all if no year is found
    filtered_links = []
    for link in links:
        match = year_pattern.search(link)
        if match:
            year = int(match.group())
            if start_year <= year <= end_year:
                filtered_links.append(link)
        else:
            # If no year is found, assume it's valid and add it to the list
            filtered_links.append(link)

    # Download each file using wget
    for link in filtered_links:
        file_url = link if link.startswith('http') else requests.compat.urljoin(url, link)
        print(f"Downloading: {file_url}")
        try:
            subprocess.run(['wget', '-P', output_dir, file_url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {file_url}: {e}")

            

if __name__ == "__main__":
    # URL of the webpage containing file links
    webpage_url = "https://cirrus.ucsd.edu/~pierce/nonsplit_precip/precip/?C=N;O=A"

    # Directory to save the downloaded files
    output_directory = "/Livneh/unsplit/precipitation" #output_directory

    # Year range for filtering files
    start_year = 1980
    end_year = 1980

    # File extensions to download (e.g., '.nc')
    extensions = ['.nc']

    # Run the downloader
    download_files_within_year_range(webpage_url, output_directory, start_year, end_year, extensions)
