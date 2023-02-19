#!/bin/bash
# Get parent directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PARENT_DIR="$(dirname $SCRIPT_DIR)"

source $PARENT_DIR/scripts/parse_yaml.sh

# Build your solution
docker build --target eval -t swr.$swr_server.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-$team_name .

# Remove the stopped container
echo "Remove Stopped Containers"
if [[ $(docker ps -q -a) ]]; then
  docker rm  $(docker ps -q -a)
fi

# Remove the dangling image
echo "Remove Dangling Images"
if [[ $(docker images -qa -f 'dangling=true') ]]; then
  docker rmi $(docker images -qa -f 'dangling=true')
fi

# Build your solution
echo "Start Local Testing"
docker run -i swr.$swr_server.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-$team_name
