# Image Similarity search documentation!

## Description

Implement a system that allows users to upload an image and find similar products available on the e-commerce platform. Include filters based on product descriptions to refine results. E.g.: given a “total look picture”, extract all products in the picture.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://bucket-name/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://bucket-name/data/` to `data/`.


