import os
import sys
import time
import urllib.request
from urllib.parse import urlparse, parse_qs, unquote
from torch.hub import download_url_to_file


CHUNK_SIZE = 1638400
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'


def download_from_civitai(file_url: str, dest_dir: str, token: str, custom_basename: str = None):
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }

    # Disable automatic redirect handling
    class NoRedirection(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response
        https_response = http_response

    request = urllib.request.Request(file_url, headers=headers)
    opener = urllib.request.build_opener(NoRedirection)
    response = opener.open(request)

    if response.status in [301, 302, 303, 307, 308]:
        redirect_url = response.getheader('Location')

        # Extract filename from the redirect URL
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        content_disposition = query_params.get(
            'response-content-disposition', [None])[0]

        if content_disposition:
            filename = unquote(
                content_disposition.split('filename=')[1].strip('"')
            )
        else:
            raise Exception('Unable to determine filename')

        response = urllib.request.urlopen(redirect_url)
    elif response.status == 404:
        raise Exception('File not found')
    else:
        raise Exception('No redirect found, something went wrong')

    total_size = response.getheader('Content-Length')

    if total_size is not None:
        total_size = int(total_size)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    output_file = os.path.join(dest_dir, (custom_basename or filename))

    if os.path.isfile(output_file):
        return output_file

    with open(output_file, 'wb') as f:
        downloaded = 0
        start_time = time.time()

        while True:
            chunk_start_time = time.time()
            buffer = response.read(CHUNK_SIZE)
            chunk_end_time = time.time()

            if not buffer:
                break

            downloaded += len(buffer)
            f.write(buffer)
            chunk_time = chunk_end_time - chunk_start_time

            if chunk_time > 0:
                speed = len(buffer) / chunk_time / (1024 ** 2)  # Speed in MB/s

            if total_size is not None:
                progress = downloaded / total_size
                sys.stdout.write(
                    f'\rDownloading: {filename} [{progress*100:.2f}%] - {speed:.2f} MB/s')
                sys.stdout.flush()

    end_time = time.time()
    time_taken = end_time - start_time
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f'{int(hours)}h {int(minutes)}m {int(seconds)}s'
    elif minutes > 0:
        time_str = f'{int(minutes)}m {int(seconds)}s'
    else:
        time_str = f'{int(seconds)}s'

    sys.stdout.write('\n')
    print(f'Download completed. File saved as: {filename}')
    print(f'Downloaded in {time_str}')
    return output_file


def auto_download(path: str, dest_dir: str, civitai_token: str = None, custom_basename: str = None):
    os.makedirs(dest_dir, exist_ok=True)

    if path.startswith('https://civitai.com') and civitai_token:
        path = download_from_civitai(path, dest_dir, civitai_token, custom_basename)
    elif path.startswith('https://huggingface.co'):
        name = os.path.basename(path).replace('?download=true', '')
        dest_path = os.path.join(dest_dir, (custom_basename or name))
        if not os.path.isfile(dest_path):
            download_url_to_file(path, dest_path)
        path = dest_path

    return path
