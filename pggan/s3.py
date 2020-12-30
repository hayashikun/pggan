import os

import boto3
import logging

BucketName = "hayashikun"


def sync(local_path, obj_prefix):
    bucket = boto3.resource("s3").Bucket(BucketName)
    exist_objs = [o.key for o in bucket.objects.filter(Prefix=obj_prefix)]

    for root, dirs, files in os.walk(local_path):
        for file in files:
            if file.startswith("."):
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, local_path)
            key = os.path.join(obj_prefix, relative_path)
            if key in exist_objs:
                exist_objs.remove(key)
            else:
                upload(key, file_path)
                logging.info(f"uploaded {key}")

    for key in exist_objs:
        if key.endswith("/"):
            continue
        download(key, os.path.join(local_path, key[len(obj_prefix):]))
        logging.info(f"downloaded {key}")


def download(key, path):
    bucket = boto3.resource("s3").Bucket(BucketName)
    if path.endswith("/"):
        path = os.path.join(path, key)
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "wb") as f:
        bucket.download_fileobj(Key=key, Fileobj=f)


def upload(key, path):
    bucket = boto3.resource("s3").Bucket(BucketName)
    with open(path, "rb") as f:
        bucket.put_object(Key=key, Body=f)


def upload_directory(path, prefix=""):
    bucket = boto3.resource("s3").Bucket(BucketName)
    if not os.path.isdir(path):
        raise ValueError("This is not directory path")

    parent_path = os.path.dirname(path)

    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.startswith("."):
                dirs.remove(d)
        for file in files:
            if file.startswith("."):
                continue
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, parent_path)
            s3_path = os.path.join(prefix, relative_path)

            with open(local_path, 'rb') as f:
                bucket.put_object(Key=s3_path, Body=f)
