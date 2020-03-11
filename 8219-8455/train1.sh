#!/bin/bash
cdate=`date +%m-%d-%Y_%H-%M-%S`
docker run --rm -it --runtime=nvidia -v /home:/home -v /mnt:/mnt -w /home/alex/sat/mySAT --user 1000:1000 --name mysat01 super_puper_ml_container python3 experiments.py