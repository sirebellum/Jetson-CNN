source activate tensorflow-cpu
while :
do
python eval_cnn.py $2 --eval=1
sleep $1
done
