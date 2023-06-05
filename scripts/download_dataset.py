#!/usr/bin/python3

import os
from pathlib import Path

import yaml
from obs import ObsClient


def download_dataset(download_dir="downloads"):
    """
    Downloads the latest Dataset from the evaluation if it exists. If the latest Dataset already exists in the download
    folder, it will notify you and not download it again. Please make sure you provided the proper credentials to the
    team_info.yml, otherwise the authentication will fail.
    """
    with open(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) +
              "/air_hockey_agent/team_info.yml") as stream:
        secrets = yaml.safe_load(stream)

    AK = secrets["AK"]
    SK = secrets["SK"]
    team_name = secrets["team_name"].lower().replace(" ", "-")
    swr_server = secrets["swr_server"].lower().replace(" ", "-")

    Path(download_dir).mkdir(parents=True, exist_ok=True)

    server = 'https://obs.{}.myhuaweicloud.eu'.format(swr_server)
    bucketName = 'air-hockey-dataset-eu'
    objectKey = f'data-{team_name}.zip'
    obsClient = ObsClient(access_key_id=AK, secret_access_key=SK, server=server)

    resp = obsClient.getObjectMetadata(bucketName, objectKey)

    if resp['status'] != 200:
        raise Exception("Could not get download object: ", resp['reason'])

    last_modified = resp['header'][3][1][5:19].replace(" ", "-")

    for old_dataset in os.listdir(download_dir):
        if last_modified in old_dataset:
            print("There is no new Dataset available")
            return

    resp = obsClient.getObject(bucketName, objectKey,
                               downloadPath=os.path.join(download_dir, f"dataset-{last_modified}.zip"))

    if resp['status'] != 200:
        raise Exception("Could not get Object: ", resp['reason'])

    print(f"Successfully downloaded dataset-{last_modified}.zip")


if __name__ == "__main__":
    download_dataset()
