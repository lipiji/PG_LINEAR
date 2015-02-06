#!/bin/bash
#
# Regularized logistic regression using Gradient Decent (batch)
# Paper : Stochastic Gradient Descent Training for 
# 		  L1-regularized Log-linear Models with Cumulative Penalty
# lipiji.pz@gmail.com
# zhizhihu.com
# 2013-03-15

awk 'BEGIN{
	OFS = "\t";
	OFMT = "%.6f";
	Lp = 1; # L-p norm

	## ini
	epsilon = 0.00001;
	gamma = 0.01;
	lambda = 0.9;
	max_iters = 5000;

	n = 0;
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
	Y[n] = $NF;
	for(i=1; i<NF; i++)
		X[n, i] = $i;
	d = NF-1;
	X[n, 0] = 1; # b
}
END{
	##normalization
	for(i=0; i<=d; i++)
	{
		max_v[i] = -999999;
		for(j=1; j<=n; j++)
		{
			if(abs(X[j, i]) > max_v[i])
				max_v[i] = abs(X[j, i]);
		}
		max_v[i] = (max_v[i] == 0) ? 1:max_v[i] ;
	}
	
	for(i=1; i<=n; i++)
	{
		for(j=0; j<=d; j++)
			X[i,j] /= max_v[j]; 
	}
	
	for(i=0; i<=d; i++)
	{
		theta[i] = 0;
	}

	iter = 0;
	epsilon_i = 999;
	while(iter < max_iters && epsilon_i > epsilon)
	{
		J_theta = 0;
		for(i=1; i<=n; i++)
		{
			h_x[i] = 0;
			for(j=0; j<=d; j++)
				h_x[i] += theta[j] * X[i, j];
			h_x[i] = sigmoid(h_x[i]);

			## Loss
			J_theta += (Y[i] * log(h_x[i]) + (1 - Y[i]) * log(1 - h_x[i]));
		}

		## Regularized 
		norm_th = 0
		for(i=0; i<=d; i++)
			norm_th += abs(theta[i])

		J_theta = -1 / (2*n) * J_theta + (lambda / n) * norm_th;
		printf "Loss = %.6f,\t", J_theta;

		## update theta
		for(j=0; j<=d; j++)
		{
			grad[j] = 0;
			for(i=1; i<=n; i++)
				grad[j] += (h_x[i] - Y[i]) * X[i, j];
		
			theta_old[j] = theta[j];
			if(j == 0)
				theta[j] = theta[j] - gamma * grad[j] / n;
			else
				theta[j] = theta[j] - gamma * (grad[j] / n + lambda / n * Sign(theta[j]));
		}

		## compute the norm
		sum = 0;
		for(i=0; i<=d; i++)
			sum += (theta[i] - theta_old[i])^2;
		epsilon_i = sqrt(sum);
	
		printf "Iter = %d,\tEpsilon = %.6f.\n", iter, epsilon_i;
		iter++; 
	}
	
	for(i=0; i<=d; i++)
		printf "%.6f\t", theta[i];
	printf "\n";

	tp = 0;
	fp = 0;
	tn = 0;
	fn = 0;
	auc = 0;
	precision = 0;
	recall = 0;
	f1 = 0;
	for(i=1; i<=n; i++)
	{
		pred[i] = 0;
		plr = 0;
		for(j=0; j<=d; j++)
			plr+=theta[j]*X[i,j];
		plr = sigmoid(plr);

		pred[i] = (plr >= 0.5) ? 1:0;

		if(pred[i] == Y[i] && pred[i] == 1)
			tp++;
		if(pred[i] == Y[i] && pred[i] == 0)
			tn++;
		if(pred[i] != Y[i] && pred[i] == 1)
			fp++;
		if(pred[i] != Y[i] && pred[i] == 0)
			fn++;
	}
	precision = tp / (tp + fp);
	recall = tp / (tp + fn);
	auc = (tp + tn) / (tp +tn +fp +fn);
	f1 = 2*precision*recall / (precision + recall);

	print "Samples = "n, "Positive = "(tp + fn);
	print "Dim = "d;
		
	print "Confusion Matrix = "
	print tp, fp;
	print fn, tn;

	print "Accuracy = "auc;
	print "Precision = "precision;
	print "Recall = "recall;
	print "F1 = "f1; 

}'
