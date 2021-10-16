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
	matB = malloc(N*N*sizeof(float));
    matBnew = calloc(N*N,sizeof(float));
	matC = calloc(N*N,sizeof(float));
	
	//randomize A and B
	randomizeMatrix(R_CONST,N,matA);
	randomizeMatrix(R_CONST,N,matB);
	float tmp = 0;
	
	//get start time 
	double time;
	time = omp_get_wtime();

    //transpose matrix b
		for (j = 0; j < N; j++){
			for (k = 0; k < N; k++){
                matBnew[index(j, k, N)] = matB[index(k, j, N)];
            }
        }
	
    
	//do matrix multiply
        float val = 0;
        #pragma omp parallel for private(i, j, k)
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++){
                matC[index(i,j,N)] = 0;
                for (k = 0; k < N; k++)
                    matC[index(i,j,N)] += matA[index(i,k,N)]*matBnew[index(j,k,N)];
            }
        }
	
	//calculate elapsed time for matrix multiply
	time = omp_get_wtime() - time;
	
	//calculate metrics 
	long FLOP = 2 * pow(N,3); 
	double Flops = FLOP/time;
	
	#ifdef VERBOSE //by default not included 
	printf("Performed a %d x %d matrix multiply transpose in %f seconds\n", N, N, time);
	printf("Number of floating point operations = 2 * %d^3 = %ld\n", N, FLOP);
	printf("Flops = %e\n", Flops);
    printf("Element from Matrix C[4][5]: %f\n", matC[index(4,5,N)]);
	#endif
	
	//(optional) save metrics to file if argv[2] exists
	if(argc > 2){ //assume (in production code, check) 2nd arg is datafile out 
		FILE * csv_file;
		
		//first check if the file has already been created
		csv_file = fopen(argv[2],"r");
		if(!csv_file){
			//not created, reopen file as write, write column headers
			csv_file = fopen(argv[2],"w");
			fprintf(csv_file, "N,FLOP,Flops,s,Threads\n");
		}
		fclose(csv_file); //close outside brackets in case the fopen in read worked
		
		// file is created from here onward, open in append mode and add the data 
		csv_file = fopen(argv[2],"a");
		fprintf(csv_file, "%d,%ld,%e,%lf,%d\n",N,FLOP,Flops,time,num_threads);
		fclose(csv_file);
	}
	
	//free memory
	free(matA);
	free(matB);
	free(matC);
	
	return 0;
	
}
