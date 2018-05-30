source activate tensorflow-cpu
while :
do
python eval_cnn.py models/11x11_ww21 --eval=1
sleep $1
done
