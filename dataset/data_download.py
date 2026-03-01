import os
import tarfile
import zipfile
from tqdm import tqdm
import requests
import urllib.request

    
    
def download_tinyimagenet(root):
    """
    Download TinyImageNet dataset if it doesn't exist.
    
    Args:
        root (str): Root directory to store the dataset
    """
    if os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
        print('TinyImageNet dataset already exists.')
        return
    
    print('Downloading TinyImageNet dataset...')
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    
    # Create directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Define a custom progress bar for downloading
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    # Download the file with progress bar
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Downloading') as t:
        urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    
    # Extract the zip file with progress bar
    print('Extracting dataset...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total number of files in zip
        total_files = len(zip_ref.namelist())
        
        # Extract all files with progress bar
        for file in tqdm(zip_ref.namelist(), total=total_files, desc='Extracting'):
            zip_ref.extract(file, root)
    
    # Remove the zip file
    os.remove(zip_path)
    print('Downloaded and extracted TinyImageNet dataset.')
