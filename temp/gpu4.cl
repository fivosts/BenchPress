__kernel void 
reduce_kernel(	long d_Ne,													// number of elements in array
				long d_no,													// number of sums to reduce
				int d_mul,													// increment
				__global fp* d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
				__global fp* d_sums2,
				int gridDim){

	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	// int gridDim = (int)get_group_size(0)/(int)get_local_size(0);			// number of workgroups
	int nf = NUMBER_THREADS-(gridDim*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = 0;															// divisibility factor for the last block

	// statistical
	__local fp d_psum[NUMBER_THREADS];										// data for block calculations allocated by every block in its shared memory
	__local fp d_psum2[NUMBER_THREADS];

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads

		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];

	}

    // Lingjie Zhang modificated at Nov 1, 2015
	//	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); // Lukasz proposed, Ke modified 2015/12/12 22:31:00
    // end Lingjie Zhang modification

	// reduction of sums if all blocks are full (rare case)	
	if(nf == NUMBER_THREADS){
		// sum of every 2, 4, ..., NUMBER_THREADS elements
		for(i=2; i<=NUMBER_THREADS; i=2*i){
			// sum of elements
			if((tx+1) % i == 0){											// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
			// synchronization
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		// final sumation by last thread in every block
		if(tx==(NUMBER_THREADS-1)){											// block result stored in global memory
			d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
		}
	}
	// reduction of sums if last block is not full (common case)
	else{ 
		// for full blocks (all except for last block)
		if(bx != (gridDim - 1)){											//
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				// sum of elements
				if((tx+1) % i == 0){										// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			// final sumation by last thread in every block
			if(tx==(NUMBER_THREADS-1)){										// block result stored in global memory
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
		// for not full block (last block)
		else{																//
			// figure out divisibility
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				if(nf >= i){
					df = i;
				}
			}
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=df; i=2*i){											//
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 && tx<df){								// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization (all threads)
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			// remainder / final summation by last thread
			if(tx==(df-1)){										//
				// compute the remainder and final summation by last busy thread
				for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){						//
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
				// final sumation by last thread in every block
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
	}

}