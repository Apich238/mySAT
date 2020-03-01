#!/bin/bash
docker run --rm -it --runtime=nvidia -v /home:/home -v /mnt:/mnt -w /home/alex/sat/mySAT --name sat super_puper_ml_container python3 experiments.py