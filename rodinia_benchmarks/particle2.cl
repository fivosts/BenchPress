#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
 
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#elif defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/** added this function. was missing in original double version.
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double dev_round_double(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}


/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a double representing the sum
********************************/
double calcLikelihoodSum(__global unsigned char * I, __global int * ind, int numOnes, int index){
	double likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((double)(I[ind[index*numOnes + x]] - 100),2) - pow((double)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}
/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
*****************************/
void cdfCalc(__global double * CDF, __global double * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double d_randu(__global int * seed, int index)
{

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((double) M));
}

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double d_randn(__global int * seed, int index){
	//Box-Muller algortihm
	double pi = 3.14159265358979323846;
	double u = d_randu(seed, index);
	double v = d_randu(seed, index);
	double cosine = cos(2*pi*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
****************************/
double updateWeights(__global double * weights, __global double * likelihood, int Nparticles){
	int x;
	double sum = 0;
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
		sum += weights[x];
	}		
	return sum;
}

int findIndexBin(__global double * CDF, int beginIndex, int endIndex, double value)
{
	if(endIndex < beginIndex)
		return -1;
	int middleIndex;
	while(endIndex > beginIndex)
	{
		middleIndex = beginIndex + ((endIndex-beginIndex)/2);
		if(CDF[middleIndex] >= value)
		{
			if(middleIndex == 0)
				return middleIndex;
			else if(CDF[middleIndex-1] < value)
				return middleIndex;
			else if(CDF[middleIndex-1] == value)
			{
				while(CDF[middleIndex] == value && middleIndex >= 0)
					middleIndex--;
				middleIndex++;
				return middleIndex;
			}
		}
		if(CDF[middleIndex] > value)
			endIndex = middleIndex-1;
		else
			beginIndex = middleIndex+1;
	}
	return -1;
}


/*****************************
* CUDA Find Index Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: weights
* param8: Nparticles
*****************************/


__kernel void sum_kernel(__global double* partial_sums, int Nparticles)
{

	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);

	if(i == 0)
	{
		int x;
		double sum = 0;
		int num_blocks = ceil((double) Nparticles / (double) THREADS_PER_BLOCK);
		for (x = 0; x < num_blocks; x++) {
			sum += partial_sums[x];
		}
		partial_sums[0] = sum;
	}
}
