#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

//comment to silence statistics of performance 
#define VERBOSE

//default size of matrices are MSIZE x MSIZE
#define MSIZE 4096

//range of random numbers
#define R_CONST 100 

//macro to index into matrix at location (i,j)
#define index(i,j,rowsize) i*rowsize+j

void randomizeMatrix(float range, int N, float * matrix){
	//uniform randomly initialize values in array between -range and +range
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++){
			matrix[index(i,j,N)] = (((float)rand()/(float)RAND_MAX) * 2 * range) - range;
		}
}

static int num_threads;
int main(int argc, char *argv[])
{ 
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
	int i,j,k,N;
	
	if(argc < 2){
		printf("Not enough arguments, defaulting to N=%d. Usage: [N] [(optional) data_filename]\n",MSIZE);
		N = MSIZE;
	}else{
		//get size of matrix 
		N = atoi(argv[1]);
	}
	
	srand(1); //seed the RNG
	
	//initialize memory 
	float * matA;
	float * matB;
	float * matBnew;
	float * matC;
	
	matA = malloc(N*N*sizeof(float));
	//matB = malloc(N*N*sizeof(float));
    //matBnew = calloc(N,sizeof(float));
	//matC = calloc(N*N,sizeof(float));
	
	//randomize A and B
	randomizeMatrix(R_CONST,N,matA);
	//randomizeMatrix(R_CONST,N,matBnew);
	float tmp = 0;
	float sum1 = 0;
    float sum2 = 0;
    float mean =0;
    float variance = 0;
	//get start time 
	double time;
	time = omp_get_wtime();

	//do matrix variance
    //n = sum1 = sum2 = 0
    float a = 0;
#pragma omp parallel for private(i,j,a) reduction(+:sum1)
for (i=0; i<N; i++)
{
    for (j=0; j<N; j++)
    {
        a += matA[index(i,j,N)];   
    }
    sum1 =a;
}
    printf("sum1: %f\n",sum1);
    mean = sum1 / (N*N);
    
#pragma omp parallel for private(i,j) reduction(+:sum2)
for (i=0; i<N; i++)
{
    for (j=0; j<N; j++)
    {
        sum2 += (matA[index(i,j,N)] - mean) * (matA[index(i,j,N)] - mean);
    }
}
    
    
    variance = sum2 / ((N*N) - 1);
    
    float std_dev = sqrt(variance);
    
    
    
	//calculate elapsed time for matrix multiply
	time = omp_get_wtime() - time;
	
	//calculate metrics 
	long FLOP = 2 * pow(N,3); 
	double Flops = FLOP/time;
	
	#ifdef VERBOSE //by default not included 
    printf("The Variance is %f\n", variance);
    printf("The Standard Deviation is %f\n", std_dev);
	printf("Found the variance and Standard Deviation of a %d x %d matrix in %f seconds\n", N, N, time);
	printf("Number of floating point operations = 2 * %d^3 = %ld\n", N, FLOP);
	printf("Flops = %e\n", Flops);

	#endif
	
	//(optional) save metrics to file if argv[2] exists
	if(argc > 2){ //assume (in production code, check) 2nd arg is datafile out 
		FILE * csv_file;
		
		//first check if the file has already been created
		csv_file = fopen(argv[2],"r");
		if(!csv_file){
			//not created, reopen file as write, write column headers
			csv_file = fopen(argv[2],"w");
			fprintf(csv_file, "N,FLOP,Flops,s\n");
		}
		fclose(csv_file); //close outside brackets in case the fopen in read worked
		
		// file is created from here onward, open in append mode and add the data 
		csv_file = fopen(argv[2],"a");
		fprintf(csv_file, "%d,%ld,%e,%lf\n",N,FLOP,Flops,time);
		fclose(csv_file);
	}
	
	//free memory
	free(matA);
	free(matB);
	free(matC);
	
	return 0;
	
}
