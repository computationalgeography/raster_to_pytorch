#!/bin/bash
#SBATCH --job-name=merge_swe_year_2009-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1    # define number of cores 
#SBATCH --nodes=1

# #SBATCH --ntasks-per-node=1
# #SBATCH --threads-per-core=1     # define number of threads per core it can be 1 or 2 (if we have duale core)
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu
#SBATCH --mem=200G       # define allocated memory 
#SBATCH -o job_%x_%j.out  
#SBATCH -e job_%x_%j.err
#SBATCH --mail-user=s.farajigargari@uu.nl
#SBATCH --mail-type=ALL

# module purge


# load conda
source ~/.bashrc

# activate environment
conda activate jupyter_env

# Years to process
#years_list=(1981 1982)
#years_list=(1980 1981)

#years_list=(1980 1983 1984)
#years_list=(1982 1983 1984)

#years_list=(1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996)
#
#years_list=(1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008)
#
years_list=(2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020)


bash bash_merge_year.sh "${years_list[@]}"