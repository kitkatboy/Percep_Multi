#! /bin/bash

k=0
for ((i=10;i<=60;i++))
do
	for ((j=21;j<=30;j++))
	do
		k=$(bc -l <<< "scale=1; $j/10")
		./Reseau_neurones $i $k;
	done
done
