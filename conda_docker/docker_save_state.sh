#Any commands ran in the containter that you wish to be persisted can be placed in the commited image with this command.
#You should add any important commnads to the Dockerfile 
#If they took a long time to run then it could be benifical to commit them
docker commit gene_ml gene_ml_image_commited 