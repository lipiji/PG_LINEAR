#!/bin/bash
#
# ./RLR_Stochastic_GD_L2.sh toy.txt toy.txt
#
# Regularized logistic regression using Gradient Decent (stochastic)
# Paper : Stochastic Gradient Descent Training for 
# 		  L1-regularized Log-linear Models with Cumulative Penalty
# lipiji.sdu@gmail.com
# zhizhihu.com
# 2013-03-19 : Orz~

if [ $# -lt 2 ];
then
	echo "Parameter error: trian, test;";
	exit 1;
fi

train_dat=$1;
test_dat=$2;

max_iters=200
e=0.000001

> theta_new;
> theta_old;

for((i=1;i<="$max_iters";i++))
{

cp theta_new theta_old

awk 'BEGIN{
	OFS = "\t";
	OFMT = "%.6f";
	Lp = 2; # L-p norm

	## ini
	gamma = 0.005;
	lambda = 0.1;
	J_theta = 0;
	n = 0;

	while(getline < "theta_old")
	{
		for(i=1; i<=NF; i++)
			theta[i-1] = $i;
	}
	close("theta_old");

}
function sigmoid(a)
{
	if(a >=200) return 0;
	else if(a <= -200) return 1;
	else
		return 1 / (1 + exp(-1 * a));
}
function abs(a)
{
	if(a+0 >= 0) 
		return a;
	else 
		return -1 * a;
}
function Sign(x)
{
	if(x > 0)
		return 1;
	if(x == 0)
		return 0;
	if(x < 0)
		return -1;
}
{
	n++;
	Y[1] = $NF;
	for(i=1; i<NF; i++)
		X[1, i] = $i;
	d = NF-1;
	X[1, 0] = 1; # b

	if(NR == 1)
	{
		if(theta[0] == "")
		{
			for(i=0; i<=d; i++)
				theta[i] = 0;
		}
		else
		{
			for(i=0; i<=d; i++)
				theta_old[i] = theta[i];
		}
	}


	h_x[1] = 0;
	for(i=0; i<=d; i++)
		h_x[1] += theta[i] * X[1, i];

	## update theta
	for(j=0; j<=d; j++)
	{
		grad[j] = (h_x[1] - Y[1]) * X[1, j];		
		if(j == 0)
			theta[j] = theta[j] - gamma * grad[j];
		else
			theta[j] = theta[j] - gamma * (grad[j] + lambda * theta[j]);
	}


}
END{
	seq = theta[0];
	for(i=1; i<=d; i++)
		seq = seq"\t"theta[i];

	print seq;
	
}' $train_dat  > theta_new


awk 'BEGIN{
	OFS="\t";
	d = 0;

	J_theta = 0;
	n = 0;

	while(getline < "theta_old")
	{	
		d = NF - 1;
		for(i=1; i<=NF; i++)
			theta_old[i-1] = $i;
	}
	close("theta_old");

	while(getline < "theta_new")
	{
		d = NF - 1;
		for(i=1; i<=NF; i++)
			theta_new[i-1] = $i;
	}
	close("theta_new");

	epsilon_i = 0;
	for(i=0; i<=d; i++)
		epsilon_i += (theta_new[i] - theta_old[i])^2;
	epsilon_i = sqrt(epsilon_i);
	printf "Iter = %d,    Epsilon = %.6f    ", "'$i'", epsilon_i;
}
{
	n++;
	Y[1] = $NF;
	for(i=1; i<NF; i++)
		X[1, i] = $i;
	d = NF-1;
	X[1, 0] = 1; # b
	
	h_x[1] = 0;
	for(i=0; i<=d; i++)
		h_x[1] += theta_new[i] * X[1, i];

	#print n, h_x[1], Y[1];

	J_theta += (h_x[1] - Y[1])^2;
}END{

	printf "Loss = %.6f\n", J_theta/n;	

}' $train_dat
	
}

echo -e "Training Finished. Theta = ";
awk '{OFS="\t"; print $0;}' theta_new

echo -e "Testing..."
awk 'BEGIN{
	OFS="\t";
	d = 0;
	J_theta = 0;
	n = 0;
	while(getline < "theta_new")
	{
		d = NF - 1;
		for(i=1; i<=NF; i++)
			theta_new[i-1] = $i;
	}
	close("theta_new");

}
{
	n++;
	Y[1] = $NF;
	for(i=1; i<NF; i++)
		X[1, i] = $i;
	d = NF-1;
	X[1, 0] = 1; # b
	
	h_x[1] = 0;
	for(i=0; i<=d; i++)
		h_x[1] += theta_new[i] * X[1, i];

	J_theta += (h_x[1] - Y[1])^2;
}END{

	printf "Testing MSE(Mean Squared Error) = %.6f\n", J_theta/n;	

}' $test_dat

rm theta_*
