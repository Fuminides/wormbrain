#!/bin/bash
#$ -N Wormbrain
#$ -q cal.q
#$ -cwd
#$ -M javier.fumanal@unavarra.es
#$ -m ae
#$ -t 1

source activate py365
python ./analyze_model.py >./salida_class.txt 2>./error_class.txt
