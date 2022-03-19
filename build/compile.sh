# Configuration variables
TargetDirectories=(
    homework-practice-01/homework-practice-01-Pilipenko-Sergey.tex
    homework-practice-02/homework-practice-02-Pilipenko-Sergey.tex
    homework-theory-01/homework-theory-01-Pilipenko-Sergey.tex
    homework-theory-02/homework-theory-02-Pilipenko-Sergey.tex
)
ArtifactDirectory=artifacts/


get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}


ArtifactDirectory=$(get_abs_filename "$ArtifactDirectory")
if [ ! -d "$ArtifactDirectory" ]; then
    mkdir -p $ArtifactDirectory
fi

for directory in ${TargetDirectories[@]}; do
    if [ -f "$directory" ]; then
        latexmk -pdf -cd "$directory" -output-directory=$ArtifactDirectory
    fi
done
