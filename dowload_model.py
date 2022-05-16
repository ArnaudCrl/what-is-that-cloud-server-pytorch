import aiohttp
from pathlib import Path
import asyncio


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


model_file_url = 'https://www.dropbox.com/s/cwmnjd5s6t3puo8/what_is_that_cloud_squeezenet.pt?dl=1'
model_file_name = 'what_is_that_cloud_squeezenet.pt'
asyncio.run(download_file(model_file_url, Path(model_file_name)))

model_file_url = 'https://www.dropbox.com/s/2gitn8ac3jdr3ha/what_is_that_cloud_mobilenet_v2.pt?dl=1'
model_file_name = 'what_is_that_cloud_mobilenet_v2.pt'
asyncio.run(download_file(model_file_url, Path(model_file_name)))

model_file_url = 'https://www.dropbox.com/s/bcvy0mtade0mcxh/what_is_that_cloud_alexnet.pt?dl=1'
model_file_name = 'what_is_that_cloud_alexnet.pt'
asyncio.run(download_file(model_file_url, Path(model_file_name)))