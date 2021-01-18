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
bucketcount( global float *input, global int *indice, global uint *d_prefixoffsets, const int size, global float *l_pivotpoints)
{
    
	volatile __local uint s_offset[BUCKET_BLOCK_MEMORY];
    
    const uint threadTag = get_local_id(0) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0))
		s_offset[i] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
	for (int tid = get_global_id(0); tid < size; tid += numThreads) {
		float elem = input[tid];
        
		int idx  = DIVISIONS/2 - 1;
		int jump = DIVISIONS/4;
		float piv = l_pivotpoints[idx]; //s_pivotpoints[idx];
        
		while(jump >= 1){
			idx = (elem < piv) ? (idx - jump) : (idx + jump);
			piv = l_pivotpoints[idx]; //s_pivotpoints[idx];
			jump /= 2;
		}
		idx = (elem < piv) ? idx : (idx + 1);
        
		indice[tid] = (addOffset(s_offset + warpBase, idx, threadTag) << LOG_DIVISIONS) + idx;  //atomicInc(&offsets[idx], size + 1);
	}
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0))
		d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
}