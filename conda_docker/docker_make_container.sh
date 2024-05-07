#!/bin/sh
#Use this to create the gene_ml container and run it.
# The container will be removed and free up resources when the terminal is closed. This is because of the --rm flag
sudo docker run --name gene_ml -t -i --rm -v /home/$USER/:/home/$USER/ gene_ml_image /bin/bash

