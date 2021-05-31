// #define BLOCK_SIZE 16


__kernel void
lud_internal(__global float *m, 
			 __local  float *peri_row,
			 __local  float *peri_col,
			int matrix_dim, 
			int offset)
{
  
  int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;


}





