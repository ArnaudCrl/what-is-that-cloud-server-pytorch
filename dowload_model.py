import aiohttp
from pathlib import Path
import asyncio


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


model_file_url = 'https://www.dropbox.com/s/a74790yltd6qp68/what_is_that_cloud_ml_model.pt?dl=1'
model_file_name = 'what_is_that_cloud_ml_model.pt'
asyncio.run(download_file(model_file_url, Path(model_file_name)))