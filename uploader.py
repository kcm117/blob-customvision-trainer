import asyncio
import json
import requests

from azure.storage.blob import BlockBlobService
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

# //TODO - Clean up this mess later.  v 0.0.1

# Load keys from keys.json
with open('keys.json', 'r') as json_file:
    keys = json.load(json_file)

# Store Keys
storage_acct_name = "kcmunninstoragev2"
storage_acct_key = keys.get("storage_key")
storage_container_name = "animals"
BATCH_SIZE = 64

START_TIME = default_timer()

def get_blob_filename(blob_path):
    return blob_path.split('/')[-1]

def get_blob(session, blob_name):
    base_url = "https://{}.blob.core.windows.net/{}/".format(
        storage_acct_name,
        storage_container_name
    )
    with session.get(base_url + blob_name) as response:
        data = (response.url,response.content)
        if response.status_code != 200:
            print("FAILURE::{0}".format(session.url))
        # Now we will print how long it took to complete the operation from the 
        # `get_blob` function itself
        elapsed = default_timer() - START_TIME
        time_completed_at = "{:5.2f}s".format(elapsed)
    print("{0:<30} {1:>20}".format(blob_name, time_completed_at))
    return data

async def get_blob_asynchronous(batch_input,batch_output_list):
    print("{0:<30} {1:>20}".format("File", "Completed at"))
    with ThreadPoolExecutor(max_workers=32) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `get_blob`
            loop = asyncio.get_event_loop()
            START_TIME = default_timer()
            tasks = [
                loop.run_in_executor(
                    executor,
                    get_blob,
                    *(session, blob) # Allows us to pass in multiple arguments to `get_blob`
                )
                for blob in batch_input
            ]
            for response in await asyncio.gather(*tasks):
                customvision_image = ImageFileCreateEntry(name=get_blob_filename(response[0]),contents=response[1],tag_ids=[tags_dict.get("cat").id])
                batch_output_list.append(customvision_image)

# submit batch size to custom vision

#  Blob
block_blob_service = BlockBlobService(account_name=storage_acct_name, account_key=storage_acct_key)
generator = block_blob_service.list_blobs(storage_container_name, prefix='train/cat')
blob_list = [blob.name for blob in generator]

# batch list into sizes of 64
batches = [blob_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range((len(blob_list) + BATCH_SIZE - 1) // BATCH_SIZE )] 

# Custom Vision
cv_endpoint = 'https://southcentralus.api.cognitive.microsoft.com'
cv_projectid = keys.get("customvision_projectid")
cv_training_key = keys.get("customvision_training_key")

trainer = CustomVisionTrainingClient(cv_training_key, endpoint=cv_endpoint)
project = trainer.get_project(cv_projectid)

# trainer.create_tag(cv_projectid, "cat", description="cat", type="Regular")
# trainer.create_tag(cv_projectid, "dog", description="dog", type="Regular")

tags_dict = {tag.name:tag for tag in trainer.get_tags(cv_projectid)}


def main():

    # Pull images from blob & upload Training Batches
    for i, batch in enumerate(batches):
        
        # Current batch is a list of ImageFileCreateEntry() objects
        current_batch = []

        # Async pull images from blob
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(get_blob_asynchronous(batch,current_batch))
        loop.run_until_complete(future)


        # Upload Batch to custom vision
        print("*"*20)
        print("Uploading Batch {}.".format(i+1))
        upload_result = trainer.create_images_from_files(project.id, images=current_batch)
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            print("*"*20)
            exit(-1)
        else:
            print("Image batch {} upload successful.".format(i+1))
            elapsed = default_timer() - START_TIME
            time_completed_at = "{:5.2f}s".format(elapsed)
            print("Image batch {} completed at {}".format(i+1, time_completed_at))
            print("*"*20)

    '''
    for b in blob_batch_data:
        print(b)
    '''

main()