
pull_repo () {

    if [ $# -ne 2 ]; then
        echo "Usage: pull_repo github_dir repo_url"
        exit
    fi

    github_dir=$1
    repo_url=$2

    readarray -d / -t parts <<< $repo_url
    
    repo_name=${parts[4]}
    repo_dir="$github_dir/$repo_name"

    if [ ! -d $repo_dir ]; then
        cd $github_dir
        while [ ! -d $repo_dir ]; do
            git clone $repo_url
        done
    else
        echo "Pulling from $repo_url"
        cd $repo_dir
        git pull
    fi
}

