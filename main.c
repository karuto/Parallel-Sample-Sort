/* File:       conj_grad.c
 * Author:     Vincent Zhang
 *
 * Purpose:    A serial conjugate gradient solver program. Due to time limits,
 *             the MPI parallel version is not included in this source code.
 *
 * Compile:    gcc -g -Wall -lm -o conj_grad conj_grad.c
 * Run:        conj_grad [order] [tolerance] [iterations] 
 *                       [Optional suppress output(n)] < [file]
 *
 * Input:      A file that contains a symmetric, positive definite matrix A,  
 *             and the corresponding right hand side vector B. Preferably, each
 *             line consists of [n] elements and the [n+1] line would be the b.
 * Output:     1. The number of iterations,
 *             2. The time used by the solver (not including I/O),
 *             3. The solution to the linear system (if not suppressed),
 *             4. The norm of the residual calculated by the conjugate gradient 
 *                method, and 
 *             5. The norm of the residual calculated directly from the 
 *                definition of residual.
 *
 * Algorithm:  The matrix A's initially read and parsed into an one-dimensional
 *             array; the right hand side vector b is stored in an array as 
 *             well. After some preparation work of allocating memory and 
 *             assigning variables the program jumps into the main loop, the 
 *             conjugate gradient solver. For the exact mathematical procedure,
 *             please refer to http://www.cs.usfca.edu/~peter/cs625/prog2.pdf
 *             and http://en.wikipedia.org/wiki/Conjugate_gradient_method for a
 *             much better demonstration.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "timer.h"

#define BARRIER_COUNT 1000

// Synchronization tools
int barrier_thread_count = 0;
pthread_mutex_t barrier_mutex;
pthread_cond_t ok_to_proceed;

// Function headers
void Usage(char* prog_name);
void Print_list(int *l, int size, char *name);
int Is_used(int seed, int offset, int range);
void *Thread_work(void* rank);

// Global variables
int i, thread_count, sample_size, list_size;
int *list, *sample_keys, *sorted_keys;
char *input_file;



/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name) {

  fprintf(stderr, "Usage: %s [number of threads] [sample size] [list size] [name of input file] [Optional suppress output(n)]\n", prog_name);
  exit(0);
}  /* Usage */



/*--------------------------------------------------------------------
 * Function:    Print_list
 * Purpose:     Print list in formatted fashion
 * In arg:      l, size, name
 */
void Print_list(int *l, int size, char *name) {
    printf("\n======= ");
    printf("%s", name);
    printf(" ======= \n");
    for (i = 0; i < size; i++) {
    	  printf("%d ", l[i]);
    }
    printf("\n\n");
}  /* Print_list */



/*--------------------------------------------------------------------
 * Function:    Is_used
 * Purpose:     Check if the random seeded key is already selected in sample
 * In arg:      seed, offset, range
 */
int Is_used(int seed, int offset, int range) {
	for (int i = offset; i < (offset + range); i++) {
		if (sample_keys[i] == list[seed]) {
			return 1;
		} else {
			return 0;
		}
	}
	return 0;
} /* Is_used */



/*-------------------------------------------------------------------
 * Function:    Thread_work
 * Purpose:     Run BARRIER_COUNT barriers
 * In arg:      rank
 * Global var:  barrier
 * Return val:  Ignored
 */
void *Thread_work(void* rank) {
  long my_rank = (long) rank;
  int i, seed, index, offset, local_chunk_size, local_sample_size;

  local_chunk_size = list_size / thread_count;
  local_sample_size = sample_size / thread_count;
  
  // printf("Hi this is thread %ld, I have %d chunks and should do %d samples. \n", my_rank, local_chunk_size, local_sample_size);
  
  // Get sample keys randomly from original list
  srandom(my_rank + 1);  
  offset = my_rank * local_sample_size;
  
  for (i = offset; i < (offset + local_sample_size); i++) {
	  do {
		  // If while returns 1, you'll be repeating this
		  seed = (my_rank * local_chunk_size) + (random() % local_chunk_size);
	  } while (Is_used(seed, offset, local_sample_size));
	  // If the loop breaks (while returns 0), data is clean, assignment
	  sample_keys[i] = list[seed];
	  index = offset + i;
	  // printf("T%ld, seed = %d\n", my_rank, seed);
	  printf("T%ld, index = %d, i = %d, key = %d, LCS = %d\n\n", my_rank, index, i, list[seed], local_sample_size);
	

  }
  
  // TODO: lock to force syncing
  
  // Parallel count sort the sample keys
  for (i = offset; i < (offset + local_sample_size); i++) {
	  int mykey = sample_keys[i];
	  int myindex = 0;
	  for (int j = 0; j < sample_size; j++) {
		  if (sample_keys[j] < mykey) {
			  myindex++;
			  
		  } else if (sample_keys[j] == mykey && j < i) {
			  myindex++;
			  
		  } else {
			  
		  }
	  }
	  printf("##### P%ld Got in FINAL, offset = %d, mykey = %d, myindex = %d\n", my_rank, offset, mykey, myindex);
	  sorted_keys[myindex] = mykey;
  }
  
  
  /*
  for (i = 0; i < BARRIER_COUNT; i++) {
     pthread_mutex_lock(&barrier_mutex);
     barrier_thread_count++;
     if (barrier_thread_count == thread_count) {
       barrier_thread_count = 0;
       pthread_cond_broadcast(&ok_to_proceed);
     } else {
       // Wait unlocks mutex and puts thread to sleep.
       //    Put wait in while loop in case some other
       // event awakens thread.
       while (pthread_cond_wait(&ok_to_proceed,
                 &barrier_mutex) != 0);
       // Mutex is relocked at this point.
     }
     pthread_mutex_unlock(&barrier_mutex);
  }
  */
  
  
  
  

  return NULL;
}  /* Thread_work */




/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
  long thread;
  pthread_t* thread_handles; 
  double start, finish;

  // for (int i = 0; i < argc; ++i){
  //   printf("Command line args === argv[%d]: %s\n", i, argv[i]);
  // }  

  if (argc != 5) { 
	Usage(argv[0]);
  } else { // TODO: process optional suppress command
  	thread_count = strtol(argv[1], NULL, 10);
  	sample_size = strtol(argv[2], NULL, 10);
  	list_size = strtol(argv[3], NULL, 10);
  	input_file = argv[4];
  }

  // Allocate memory for variables
  thread_handles = malloc(thread_count*sizeof(pthread_t));
  list = malloc(list_size * sizeof(int));
  sample_keys = malloc(sample_size * sizeof(int));
  sorted_keys = malloc(sample_size * sizeof(int));
  

  pthread_mutex_init(&barrier_mutex, NULL);
  pthread_cond_init(&ok_to_proceed, NULL);
  

  // Read list content from input
  FILE *fp = fopen(input_file, "r+");
  for (i = 0; i < list_size; i++) {
  	  if (!fscanf(fp, "%d", &list[i])) {
    	  break;
      }
  }
  Print_list(list, list_size, "original list");
  
  GET_TIME(start);
  
  for (thread = 0; thread < thread_count; thread++)
     pthread_create(&thread_handles[thread], NULL,
         Thread_work, (void*) thread);

  for (thread = 0; thread < thread_count; thread++) 
     pthread_join(thread_handles[thread], NULL);
  
  GET_TIME(finish);
  
  Print_list(sample_keys, sample_size, "sample keys (unsorted)");
  Print_list(sorted_keys, sample_size, "sample keys (sorted)");
  
  // printf("Elapsed time = %e seconds\n", finish - start);


  pthread_mutex_destroy(&barrier_mutex);
  pthread_cond_destroy(&ok_to_proceed);
  free(thread_handles);
  
  return 0;
}  /* main */


