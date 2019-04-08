import asyncio
import json
import multiprocessing
import requests
import sys

from azure.storage.blob import BlockBlobService
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

class CustomVisionBlobUploader(object):
    
    _BATCH_SIZE = 64
    _WORKER_CONCURRENCY = multiprocessing.cpu_count() * 8

    def __init__(
        self,
        storage_acct_name, storage_acct_key, storage_container_name, storage_prefix,
        cv_endpoint, cv_projectid, cv_training_key, tags
    ):
        # Configure Blob Storage Instance
        self._storage_acct_name = storage_acct_name
        self._storage_acct_key = storage_acct_key
        self._storage_container_name = storage_container_name
        self._block_blob_service = BlockBlobService(account_name=storage_acct_name, account_key=storage_acct_key)

        # Configure Custom Vision Instance
        self._cv_endpoint = "https://" + str(cv_endpoint) + ".api.cognitive.microsoft.com"
        self._cv_projectid = cv_projectid
        self._cv_training_key = cv_training_key
        self.trainer = CustomVisionTrainingClient(cv_training_key, endpoint=self._cv_endpoint)
        self.project = self.trainer.get_project(cv_projectid)

        # Handle tags that may not exist, get tags from project
        cv_tag_names = [tag.name for tag in self.trainer.get_tags(cv_projectid)]
        for tag in tags:
            if tag not in cv_tag_names:
                self.trainer.create_tag(cv_projectid, tag, description=tag, type="Regular")
        self.tags_dict = {tag.name:tag for tag in self.trainer.get_tags(cv_projectid)}

    def start_timer(self):
        self.START_TIME = default_timer()

    def load_blob_batches(self, prefix):
        generator = self._block_blob_service.list_blobs(self._storage_container_name, prefix=prefix)
        blob_list = [blob.name for blob in generator]
        batches = [blob_list[i * self._BATCH_SIZE:(i + 1) * self._BATCH_SIZE] for i in range((len(blob_list) + self._BATCH_SIZE - 1) // self._BATCH_SIZE )]
        self.batches = batches

    def get_blob_filename(self,blob_path):
        return blob_path.split('/')[-1]

    def get_blob(self, bbs, blob_name):
        blob = bbs.get_blob_to_bytes(self._storage_container_name, blob_name)
        # print elapsed time
        elapsed = default_timer() - self.START_TIME
        time_completed_at = "{:5.2f}s".format(elapsed)
        print("{0:<30} {1:>20}".format(blob_name, time_completed_at))
        return blob

    async def get_blob_asynchronous(self,batch_input,batch_output_list, tags):
        print("{0:<30} {1:>20}".format("File", "Completed at"))
        with ThreadPoolExecutor(max_workers=self._WORKER_CONCURRENCY) as executor:

            loop = asyncio.get_event_loop()
            # self.START_TIME = default_timer()
            tasks = [
                loop.run_in_executor(
                    executor,
                    self.get_blob,
                    *(self._block_blob_service, blob)
                )
                for blob in batch_input
            ]
            for response in await asyncio.gather(*tasks):
                customvision_image = ImageFileCreateEntry(name=self.get_blob_filename(response.name),contents=response.content,tag_ids=[self.tags_dict.get(tag).id for tag in tags])
                batch_output_list.append(customvision_image)


def main():
    
    # Get command line argsendpoint
    storage_acct, storage_container, storage_prefix, tags, cvendpoint = [arg for arg in sys.argv[1:]]
    tags = tags.split(",")

    print("*****Custom Vision Service - Blob Upload*****")
    
    # Load keys from keys.json
    print("Loading Keys.")
    with open('keys.json', 'r') as json_file:
        keys = json.load(json_file)

    Uploader = CustomVisionBlobUploader(storage_acct, keys.get("storage_key"), storage_container, storage_prefix,
        cvendpoint, keys.get("customvision_projectid"), keys.get("customvision_training_key"), tags)
    
    # Get batch of blob paths
    Uploader.load_blob_batches(storage_prefix)
    
    # Pull images from blob & upload Training Batches
    print("Begin Batches")
    Uploader.start_timer()
    for i, batch in enumerate(Uploader.batches):
        print("*****Batch {}*****".format(i+1))
        # Current batch is a list of ImageFileCreateEntry() objects
        current_batch = []

        # Async pull images from blob
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(Uploader.get_blob_asynchronous(batch,current_batch,tags))
        loop.run_until_complete(future)

        # Upload Batch to custom vision
        print("*"*20)
        print("Uploading Batch {}.".format(i+1))
        upload_result = Uploader.trainer.create_images_from_files(Uploader.project.id, images=current_batch)
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            print("*"*20)
            exit(-1)
        else:
            print("Image batch {} uploaded successfully.".format(i+1))
            elapsed = default_timer() - Uploader.START_TIME
            time_completed_at = "{:5.2f}s".format(elapsed)
            print("Image batch {} completed at {}".format(i+1, time_completed_at))
            print("*"*20)


if __name__ == "__main__":
  main()
