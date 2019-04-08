# Blob Storage to Custom Vision Uploader

Note: Early version, not fully tested.

This script is designed to pull images directly from a blob storage account and upload training images to the Custom Vision Service in batches.  This is beneficial if you would like to upload images directly from Azure Blob Storage to Custom Vision Service, without having them on a publicly accessible URL or dowenloading them to local storage.  You can run this script from a local machine, from an Azure VM, or anywhere you can execute Python Code and have internet connectivity!

Initial tests:
- ~ 8 seconds to pull and upload 100 tagged images from Azure Blob Storage to Custom Vision Service. (Home PC)
- ~45 seconds to pull and upload 1,000 tagged images from Azure Blob Storage to Custom Vision Service. (Azure VM)

Instructions:

1) **Keys** - Create a **keys.json file** in the same directory as the script.  You can use the keys_sample.json file as a template.  This local file will contain your Azure Storage key, Custom Vision training key, and Custom Vision project id (which can be found in the project settings page).

    ```
    {
    "storage_key":"<YOUR STORAGE ACCOUNT KEY HERE>",
    "customvision_projectid":"<YOUR CUSTOM VISION PROJECT ID HERE>",
    "customvision_training_key":"<YOUR CUSTOM VISION TRAINING KEY HERE>"
    }
    ```

2) **Training Data** - The script expects data to be structured in the blob storage account into different directories, with each directory containing files with specific tag(s).  Examples with cats and dogs classification:
    
    - container_name/animals/train/cat/many_pictures_of_cats.jpg
    - container_name/animals/train/dog/many_pictures_of_dogs.jpg

3) **Script Execution**:
    
    - Python Environment: Script was developed in Python 3.7.  
        ```
        # Install libraries from the requirements.txt file
        pip install -r requirements.txt
        
        # Or install the libraries directly in your current Python environment.
        pip install azure-storage-blob azure-cognitiveservices-vision-customvision
        ```

    - The script expects the following arguments:

        ```
        customvisionblobuploader.py <storage account name> <container name> <blob prefix> <comma separated tags> <region>
        ```

        - storage account name: the name of your storage account
        - container name:  the name of the container in your storage account which contains training images
        - blob prefix: the prefix of the blobs to be uploaded.  E.g. if the training images for cats are in **container_name/train/cat**, then you put put **"train/cat"** as the blob prefix. 
        - tags (comma separated): these are the tags you desire in the Custom Vision Service.  Tags will be created in the project if they do not exist.
        - region: the azure region in which your custom vision service is deployed.  examples: eastus2, southcentralus, northcentralus, etc.

    - Example usage:

        ```
        customvisionblobuploader.py kcmunninstoragev2 animals train/cat cat southcentralus
        ```

References:
- (Async requests in Python) https://hackernoon.com/how-to-run-asynchronous-web-requests-in-parallel-with-python-3-5-without-aiohttp-264dc0f8546
