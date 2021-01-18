#define DIVISIONS               (1 << 10)
#define LOG_DIVISIONS	(10)
#define BUCKET_WARP_LOG_SIZE	(5)
#define BUCKET_WARP_N			(1)
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				(128)


int addOffset(volatile __local uint *s_offset, uint data, uint threadTag){
    uint count;

    do{
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    }while(s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__kernel void
bucketsort(global float *input, global int *indice, __global float *output, const int size, global uint *d_prefixoffsets,
		   global uint *l_offsets)
{
	volatile __local unsigned int s_offset[BUCKET_BLOCK_MEMORY];
    
	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);
    
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0)){
		s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (int tid = get_global_id(0); tid < size; tid += numThreads) {
       
		float elem = input[tid];
		int id = indice[tid];
		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
//        if(test == 2) {
//            printf("EDLLAWD %f", elem);
//        }
	}
}
