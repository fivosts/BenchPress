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

__kernel void bucketprefixoffset(global uint *d_prefixoffsets, global uint *d_offsets, const int blocks) {
	int tid = get_global_id(0);
	int size = blocks * BUCKET_BLOCK_MEMORY;
	int sum = 0;
    
	for (int i = tid; i < size; i += DIVISIONS) {
		int x = d_prefixoffsets[i];
		d_prefixoffsets[i] = sum;
		sum += x;
	}
    
	d_offsets[tid] = sum;
}
