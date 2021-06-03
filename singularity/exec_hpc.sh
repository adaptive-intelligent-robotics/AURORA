#!/bin/bash
set -e
trap 'exit 130' INT

usage()
{
  echo "Usage:"
  echo "    ./exec_hpc -u <username> -t <gitlab_personal_token> -e <name_experiment> \n"
  echo "    Values possible for <name_experiment>:"
  echo "        - maze"
  echo "        - hexapod_camera_vertical"
  echo "        - air_hockey"
}

while getopts hsou:t:e: FLAG
do
    case "${FLAG}" in
        h)
          usage
          exit 0
          ;;
        u) username_hpc=${OPTARG};;
        t) gitlab_personal_token=${OPTARG};;
        e) exp=${OPTARG};;
        s) send_only="yes";;
        o) optimise_builds="yes";;
    esac
done

if [ -z "${gitlab_personal_token}" ] || [ -z "${exp}" ]
then
  usage 1>&2
  exit 1
fi

# Quick connection at the beginning just to check if the master multiplexer ssh connection is still running
# If not running anymore -> reconnecting directly using the following lines
ssh HPC /bin/bash <<- EOF
EOF

if [ ! -z "${optimise_builds}" ]
then
  setup_args=" --chosen ${exp} "
fi

# Building Container
SINGULARITYENV_SETUP_ARGS="${setup_args}" ./build_final_image --path-def singularity.def --commit-ref master --project aurora --personal-token "${gitlab_personal_token}" --debug

# finds the last image (in alphabetic order)
pattern="./final_aurora_20*" 
files=( $pattern )
echo "Using singularity container ${files[-1]}"

container=${files[-1]}
folder_container=${container%.sif}

folder_gitlab_notebook_experiments="/rds/general/user/${username_hpc}/home/gitlab_notebook_experiments/"
folder_ephemeral="/rds/general/user/${username_hpc}/ephemeral/"

# Sending last image to HPC
echo "Launching: 'scp ${container} ${username_hpc}@login.hpc.ic.ac.uk:${folder_gitlab_notebook_experiments}'"
scp ${container} HPC:${folder_gitlab_notebook_experiments}

if [ -z "${send_only}" ]
then
# Generate job scripts on HPC and submit them
ssh HPC /bin/bash <<- EOF
    set -e
    cd "${folder_gitlab_notebook_experiments}"
    export SINGULARITYENV_PERSONAL_TOKEN="${gitlab_personal_token}"

    singularity run --app gen_job_scripts "${container}" "/git/sferes2/exp/aurora/hpc_config_${exp}.json" "${exp}" "${container}"

    mv -v "${folder_gitlab_notebook_experiments}/${folder_container}" "${folder_ephemeral}"
    ln -s "${folder_ephemeral}/${folder_container}" "${folder_gitlab_notebook_experiments}"/"${folder_container}"

    ./exec.sh
EOF
fi

