
getYoloLabelsFirstNumAndCheck(){

_targetVal='1'

find ./labels/ -name "*.txt" -print0 | xargs -0 awk -v awk_var="$_targetVal" '
{
  if ($1 != awk_var) {
    print FILENAME ":line" NR ": " $0 ",target:" awk_var
  }
}'
}
getYoloLabelsFirstNumAndCheck;
