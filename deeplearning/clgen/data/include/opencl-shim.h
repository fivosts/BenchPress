// A shim header providing common definitions.
//
// Coarse grained control is provided over what is defined using include guards.
// To prevent the definition of unsupported storage classes and qualifiers:
//   -DCLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS
// To prevent the definition of common types:
//   -DCLGEN_OPENCL_SHIM_NO_COMMON_TYPES
// To prevent the definition of common constants:
//   -DCLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS
//
// Copyright (c) 2016-2020 Chris Cummins.
//
// clgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clgen.  If not, see <https://www.gnu.org/licenses/>.

// Unsupported OpenCL storage classes and qualifiers.
#ifndef CLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS
#define static
#define generic
#define AS
#endif  // CLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS

// Common types.
#ifndef CLGEN_OPENCL_SHIM_NO_COMMON_TYPES
#define CONVT float
#define DATA_TYPE float
#define DATA_TYP4 float4
#define DATATYPE float
#define FLOAT_T float
#define FLOAT_TYPE float
#define FPTYPE float
#define hmc_float float
#define inType float
#define outType float
#define real float
#define REAL float
#define Ty float
#define TyOut float
#define TYPE float
#define VALTYPE float
#define VALUE_TYPE float
#define VECTYPE float
#define WORKTYPE float
#define hmc_complex float2
#define mixed2 float2
#define real2 float2
#define REAL2 float2
#define mixed3 float3
#define real3 float3
#define REAL3 float3
#define FPVECTYPE float4
#define mixed4 float4
#define real4 float4
#define REAL4 float4
#define T4 float4
#define BITMAP_INDEX_TYPE int
#define INDEX_TYPE int
#define Ix int
#define KParam int
#define Tp int
#define Pixel int3
#define bool2 bool
#define Dtype float
#define half float
#define half2 float2
#define half3 float3
#define half4 float4
#define half8 float8
#define half16 float16
#define sph_u64 unsigned long
#define sph_u32 unsigned int
#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8
#define FLOAT4 float4
#define CL_DTYPE float
#define CL_DTYPE4 float4
#define KERNEL kernel void
#define atomicAdd atomic_add
#define channel
// GLib types
#define gboolean int
#define gchar char
#define guchar unsigned char
#define gint8 char
#define guint8 unsigned char
#define gushort unsigned short
#define gint16 short
#define guint16 unsigned short
#define gint int
#define guint unsigned int
#define gint32 int
#define guint32 unsigned int
#define glong long
#define gssize long
#define gulong unsigned long
#define gint64 long
#define guint64 unsigned long
#define gfloat float
#define gdouble double
#define guintptr unsigned long*
#define goffset long
#define gintptr long*
#define gpointer void*
#define gconstpointer const void*
// #define KERNEL_ARG_DTYPE int
#endif  // CLGEN_OPENCL_SHIM_NO_COMMON_TYPES

// Common constants
#ifndef CLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS
#define ACCESSES 16
#define AVER 2
#define BETA 0.5
#define BINS_PER_BLOCK 8
#define BITMAP_SIZE 1024
#define BLACK 0
#define BLK_X 8
#define BLK_Y 8
#define BLOCK 32
#define BLOCK_DIM 2
#define BLOCK_SIZE 64
#define BLOCK_SIZE_WITH_PAD 64
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCKSNUM 64
#define BUCKETS 8
#define CHARS 16
#define CLASS 'A'  // Used in npb-3.3
#define COLS 64
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_HALO_STEPS 1
#define COLUMNS_RESULT_STEPS 4
#define CONCURRENT_THREADS 128
#define CUTOFF_VAL 0.5
#define DEF_DIM 3
#define DIAMETER 16
#define DIM 3
#define DIMX 128
#define DIMY 64
#define DIMZ 8
#define DIRECTIONS 4
#define DISPATCH_SIZE 64
#define ELEMENTS 1024
#define EPSILON 0.5
#define EUCLID 1
#define EXPRESSION
#define EXTRA 4
#define FILTER_LENGTH 128
#define FLEN 100
#define FOCALLENGTH 100
#define FORCE_WORK_GROUP_SIZE 32
#define GAUSS_RADIUS 5
#define GLOBALSIZE_LOG2 10
#define GROUP 128
#define GROUPSIZE 128
#define HEIGHT 128
#define HISTOGRAM64_WORKGROUP_SIZE 64
#define IMAGEH 512
#define IMAGEW 1024
#define INITLUT 0
#define INITSTEP 0
#define INPUT_WIDTH 256
#define INVERSE -1
#define ITERATIONS 1000
#define KAPPA 4
#define KERNEL_RADIUS 8
#define KEY 8
#define KITERSNUM 64
#define KVERSION 1
#define LAMBDA .5
#define LENGTH 1024
#define LIGHTBUFFERDIM 128
#define LOCAL_H 8
#define LOCAL_MEM_SIZE 2048
#define LOCAL_MEMORY_BANKS 16
#define LOCAL_SIZE 128
#define LOCAL_SIZE_LIMIT 1024
#define LOCAL_W 128
#define LOCALSIZE_LOG2 5
#define LOG2_WARP_SIZE 5
#define LOOKUP_GAP 16
#define LROWS 16
#define LSIZE 64  // Used in npb-3.3
#define LUTSIZE 1024
#define LUTSIZE_LOG2 10
#define LWS 128
#define MASS 100
#define MAX 100
#define MAX_PARTITION_SIZE 1024
#define MAXWORKX 8
#define MAXWORKY 8
#define MERGE_WORKGROUP_SIZE 32
#define MOD 16
#define MT_RNG_COUNT 8
#define MULT 4
#define N_CELL_ENTRIES 128
#define N_GP 8
#define N_PER_THREAD 16
#define NCHAINLENGTH 100
#define NDIM 4
#define NEEDMEAN 1
#define NMIXTURES
#define NROUNDS 100
#define NSIEVESIZE 64
#define NSPACE 512
#define NSPIN 8
#define NTIME 100
#define NUM_OF_THREADS 1024
#define NUMBER_THREADS 32
#define OFFSET 2
#define ONE 1
#define OP_WARPSIZE 32
#define PADDING 8
#define PADDINGX 4
#define PADDINGY 2
#define PI 3.14
#define PRESCAN_THREADS 128 /* Used in parboil-0.2 histo */
#define PULSELOCALOFFSET 8
#define PULSEOFF 16
#define QPEX 1
#define QUEUE_SIZE 128
#define RADIUS 8
#define RADIUSX 16
#define RADIUSY 16
#define RADIX 2
#define REGION_WIDTH 16
#define RESULT_SIZE 512
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_HALO_STEPS 1
#define ROWS_RESULT_STEPS 4
#define ROWSIZE 128
#define SAT .5
#define SCALE
#define SCREENHEIGHT 1920
#define SCREENWIDTH 1080
#define SHAREMASK 0
#define SHA512M_A 32;
#define SHA512M_B 32;
#define SHA512M_C 32;
#define SHA512M_D 32;
#define SHA512M_E 32;
#define SHA512M_F 32;
#define SHA512M_G 32;
#define SHA512M_H 32;
#define SIGN 1
#define SIMD_WIDTH 32
#define SIMDROWSIZE
#define SINGLE_PRECISION 1
#define SIZE 1024
#define SLICE 32
#define STACK_SIZE 1024
#define STEP 8
#define SUBWAVE_SIZE 32
#define TDIR 1
#define THREADBUNCH 32
#define THREADS 2048
#define THREADS_H 16
#define THREADS_W 128
#define THREADS_X 128
#define THREADS_Y 16
#define THRESHOLD 0.5
#define TILE_COLS 16
#define TILE_COLS 16
#define TILE_DIM 16
#define TILE_HEIGHT 16
#define TILE_M 16
#define TILE_N 16
#define TILE_ROWS 16
#define TILE_SIZE 16
#define TILE_TB_HEIGHT 16
#define TILE_WIDTH 16
#define TILEH 16
#define TILESH 64
#define TILESW 16
#define TILEW 16
#define TRANSPOSEX 1
#define TRANSPOSEY -1
#define TREE_DEPTH 3
#define TX 8
#define TY 16
#define UNROLL 8
#define VECSIZE 128
#define VOLSPACE 4
#define WARP_COUNT 8
#define WARPS_PER_GROUP 8
#define WDEPTH 16
#define WDIM 2
#define WG_H 8
#define WG_SIZE 128
#define WG_SIZE_X 32
#define WG_SIZE_Y 8
#define WG_W 32
#define WIDTH 16
#define WINDOW 16
#define WORK_GROUP_SIZE 256
#define WORK_ITEMS 128
#define WORKGROUP_SIZE 256
#define WORKGROUPSIZE 256
#define WORKSIZE 128
#define WSIZE 1024
#define XDIR 0
#define XSIZE 128
#define YDIR 1
#define YSIZE 64
#define ZDIR 2
#define ZERO 0
#define ZSIZE 128
// Hashcat constants
#define KERNEL_STATIC
#define VECT_SIZE 1
#define DGST_R0 0
#define DGST_R1 1
#define DGST_R2 2
#define DGST_R3 3
#define DGST_ELEM 4
#define FIXED_LOCAL_SIZE 1
#define SCRYPT_R 8
#define DIGESTS_OFFSET 0
#define SALT_POS 0
#define SCRYPT_N 16384
#define SCRYPT_TMTO 1
#define SCRYPT_P 1
// Other constants
#define MLO_GRP_SZ0 32
#define MLO_GRP_SZ1 32
#define MLO_GRP_SZ2 32
#define MLO_GRP_SZ 0
#define MLO_N_IN_CHNLS 3
#define MLO_LCL_N_IN_CHNLS 1
#define MLO_IN_WIDTH 64
#define MLO_IN_HEIGHT 32
#define MLO_IN_STRIDE 1
#define MLO_IN_CHNL_STRIDE 1
#define MLO_IN_BATCH_STRIDE 1
#define MLO_BATCH_SZ 16
#define MLO_FLTR_SZ0 32
#define MLO_FLTR_PAD_SZ0 2
#define MLO_FLTR_STRIDE0 1
#define MLO_FLTR_SZ1 32
#define MLO_FLTR_PAD_SZ1 2
#define MLO_FLTR_STRIDE1 1
#define MLO_IN_CHNL_LOOP 1
#define MLO_OUT_WIDTH 64
#define MLO_OUT_HEIGHT 32
#define MLO_OUT_STRIDE 1
#define MLO_N_OUT_PIX_SZ0 4
#define MLO_N_OUT_PIX_SZ1 4
#define MLO_N_IN_PIX_SZ0 4
#define MLO_N_IN_PIX_SZ1 4
#define MLO_N_STACKS 1
#define MLO_N_PROCS1 8
#define MLO_N_PROCS0 8
#define MLO_IN_SZ0 0
#define MLO_IN_SZ1 1
#define BITS_GLOBAL 2
#define N_GP 4
#define PARAM_SEED 0
#define PARAM_IMAGE_WIDTH 64
#define PARAM_IMAGE_HEIGHT 64
#define PARAM_PATH_COUNT 16
#define PARAM_STARTLINE 0
#define M_PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define INV_TWOPI 0.15915494309189533577f
#define MAT_MATTE 0
#define MAT_MIRROR 1
#define MAT_GLASS 2
#define MAT_METAL 3
#define MAT_ALLOY 4
#define DOT_SIZE 128
#define NUM_ELEMENTS 64
#define NUM_FREQUENCIES 4
#define BASE_ACCUM 1
#define KB 2
#define MAX_VALUE 64
#define MIN_VALUE 0

// System configs
#define VLIW1
#define __OPENCL_VERSION__ 2
#define _CL_NOINLINE
#define _CL_ALWAYSINLINE
#define _CL_OVERLOADABLE
#define OPENCL_FORCE_INLINE
#define MIOPEN_USE_FP32 1
#endif  // CLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS

// Hacks!
// #ifndef HACKS
#define GRID_STRIDE_SHIFT_Z 2
#define GRID_STRIDE_SHIFT_Y 2
#define block_size_x 2048
#define block_size_y 1024
#define DX 16
#define DY 32
#define MAX_STACK_DEPTH 1024
#define MAX_K 32
#define ALLOW_SELF_MATCH 1
#define SORT_RESULTS 1
#define POINT_STRIDE 4
#define DIM_COUNT 3
#define SQRT_1_2_F 16
#define SQRT_1_3_F 16
#define SQRT_2_3_F 16
#define SQRT_3_4_F 16
#define SQRT_3_F 16
#define EXIT_FAILURE 1
#define LX_STAGE_21_WG_SIZE 1024
#define LX_STAGE_21_WG_PER_VECTOR 32
#define N3 16
#define lambda 0.73
#define MAGMA_S_ZERO 0
#define MAGMA_S_ONE 1
#define dev_sampler_t sampler_t
#define dev_image_t image_t
#define fp float
// #endif

#define CALbyte char
#define DCT_INT int
#define jvector_t int8
#define svm_pointer_t unsigned int
#define real4_t float4
#define ivector_t long3
#define int64 long int
#define swift_uint8_t unsigned char
#define myImage image1d_t
#define morton unsigned long long
#define swift_int64_t long
// #define *EdmaMgr_Handle void
#define VoxelId int3
#define u8x unsigned char
#define u16x unsigned short
#define hh_float8 float8
#define DPTYPE char
#define Byte unsigned char
#define bytecode unsigned long long
#define Move unsigned int
#define CALint int
#define vector_t float8
#define JCOEF short
#define JBLOCK[64] short
// #define *JBLOCKROW JBLOCK
// #define *JBLOCKARRAY JBLOCKROW
// #define *JBLOCKIMAGE JBLOCKARRAY
#define center_index_t unsigned char
#define u8_v uchar2
#define iclblasSideMode_t int
#define MACTYPE int
#define tok_t uint8_t
#define Cr unsigned long long
#define sumtype int
#define JDIMENSION unsigned int
#define wcl_uint unsigned int
#define hh_float2 float2
#define MatrixElement double
#define clo_statetype uint4
#define wcl_int int
#define BYTE unsigned char
// #define *voidpf void
#define IntPoint int4
#define cl_mem_fence_flags unsigned int
#define TvoxelValue float
#define Quaternion float4
#define myunsignedint unsigned int
#define wcl_ulong unsigned long long
#define swift_int32_t int
#define tinfl_bit_buf_t unsigned long long
#define u_int32_t unsigned int
#define uLongf unsigned long
#define point4 float4
#define wcl_unsigned_char unsigned char
#define complex_t float2
#define Indice unsigned int
#define mat4x4 float16
#define Point float4
#define s16 short
#define ElemT int2
#define wcl_short short
#define cl_ubyte32 unsigned int
#define uint64 unsigned long long
#define uchar1 unsigned char
#define real16_t float16
#define edge_index_t unsigned int
#define jscalar_t long
#define quaternion float4
#define swift_uint32_t unsigned int
#define wcl_short4 short4
#define u16a unsigned short
#define scalar_t float
#define vecto3 float3
#define chromosome int
#define real8_t float8
#define u64a unsigned long long
#define vtype unsigned int
#define ValT int
#define wcl_uchar unsigned char
#define aes_mode unsigned
#define b3Float4 float4
#define mz_bool int
#define result_type int
#define subcell_id char
#define b3Scalar float
#define wcl_long4 long4
#define uint1 unsigned int
#define distance_type int
#define WOImage write_only image1d_t
# define wcl_uint4 uint4
#define mz_uint16 unsigned short
#define index_t unsigned int
#define TsdfType int8_t
#define float1 float
#define cl_byte64 long long
#define ushort1 unsigned short
#define int8_t signed char
#define kernel_type_t const unsigned
#define keypoint float4
#define WORD unsigned int
#define svm_precision double
#define PAUChar unsigned char
#define swift_int8_t char
#define Bytef unsigned char
#define Stokes float4
#define packet uint2
#define sReal float
#define mask_t char
#define Lamp int
#define FixedPoint0 int
#define KeyValuePair uint2
#define float32_t float
#define cpx float2
#define coord_type int
#define scalar float
#define OutCode int
#define wcl_float4 float4
#define mz_uint unsigned int
#define uInt unsigned int
#define real3_t float3
#define u64 unsigned long long
#define dType float
#define center_set_pointer_t unsigned int
#define rkf_evaluation_points float4
#define des_vector unsigned int
#define Scalar float
#define magmaFloatComplex float2
#define wcl_uchar4 uchar4
#define kscalar_t int
#define u16 unsigned short
#define int_t int
#define ptype float4
#define TdetValue float
#define int32_t int
#define real2_t float2
#define bus_t uint2
#define BOOL int
#define T2 float2
#define char1 char
#define pipetype3 write_only pipe int
#define bignum25519[10] unsigned int
#define tfloat double
#define NodeId unsigned int
#define i32 int
#define int32 int
#define wcl_ushort unsigned short
#define fcomplex2 float4
#define data_t uint4
#define char_complex2 char4
#define wcl_int4 int4
#define Scalar4 float4
// #define *string constant char
#define word short
#define cl_byte32 int
#define u8 unsigned char
#define color float3
#define ScoreType int
#define node_index_t unsigned int
#define cl_ubyte8 unsigned char
#define wcl_unsigned_short unsigned short
#define fcomplex float2
#define vec2 float2
#define point2 float2
#define long1 long
#define Tree[] const int8_t
#define Hash unsigned long long
#define step_t int32_t
#define wcl_sampler_t sampler_t
#define GeglRandom ushort4
#define FixedPoint3 int
#define particle float4
// #define *voidp void
#define param_float float
#define longword long
#define xorshift6432star_state unsigned long long
#define SceneRayType int
#define cl_byte8 char
#define iclblasFillMode_t int
#define uint unsigned int
#define point3 float3
#define PAFloat float
#define FAST_FLOAT float
#define intf int
#define brief_descr int4
#define swift_uint64_t unsigned long long
#define KeyT int
#define RWImage read_write image1d_t
#define sqsumtype float
#define BitmapBuffer unsigned char
#define wcl_short4 ushort4
#define Tcoord_dev float
#define Prob uint8_t
#define vector2 float2
#define mat22 float4
#define mz_int16 signed short
#define MatrixElement float
#define NNType float
#define iclblasOperation_t int
#define Square unsigned char
#define wcl_unsigned_long unsigned long
#define atmi_taskgroup_handle_t unsigned long
#define b3Quat float4
#define index_diff_t int
#define vT float
#define jscalar_t int
#define pcg6432_state unsigned long
#define TTMove unsigned int
#define wcl_char4 char4
#define coordinate float4
#define ull unsigned long long
#define u8_v unsigned char
#define value_type int
#define FLOAT_MULT_TYPE float
#define NumberType long
#define byte unsigned char
#define T8 float8
#define charf char
#define u32 unsigned int
#define u32a unsigned int
#define random_state unsigned int
#define bc_trit_t long
#define atmi_task_handle_t unsigned long
#define wcl_float float
#define mat3x3[3][3] float
#define fcomplex4 float8
#define CALParameterb char
#define bignum25519align16[12] unsigned int
#define RandomBuffer float
#define mat3x3_tri[6] float
#define short_complex2 short4
#define fsm_state unsigned char
#define pixelType char4
#define WeightType unsigned char
#define vtype unsigned int
#define short1 short
#define fixed8 unsigned char
#define s8 char
#define DCT_FLOAT float
#define u64x unsigned long long
#define swift_uint16_t ushort
#define ulong1 unsigned long long
#define typedef_type float4
#define b3Vector3 float4
#define hh_float float
#define INT32 unsigned int
#define level_t int8_t
#define CALParameterr double
#define vector4 float4
#define swift_int16_t short
#define wcl_char char
#define RT_Vec2f float2
#define fp_t double
#define byte_t unsigned char
#define rkf_integrand_values float4
#define RT_Vec4f float4
#define PrimitiveXYIdBuffer int4
#define global_cell_id int
#define MyPipe read_only pipe int
#define string constant char*
#define vec4 float4
#define ECLPtr size_t
#define cl_byte16 short
#define uintptr_t unsigned int
#define CALParameteri int
#define intvec int16
#define state_t uint16
#define floatN float
#define realMD float
#define real_t float
#define INT16 short
#define cl_ubyte64 unsigned long long
#define CONVTYPE short
#define v4sf float4
#define RT_Vec3f float3
#define buf_t uint16
#define particle_counter int
#define RandType unsigned long long
#define PAInt int
#define mz_uint32 unsigned int
#define u32x unsigned int 
#define xorshift1024_state unsigned int
#define ROImage read_only image1d_t
#define DS float2
#define intptr_t signed int
#define realND float
#define mz_uint8 unsigned char
#define double1 double
#define iclblasDiagType_t int
#define u8a unsigned char
#define bn_word unsigned int
#define s32 int
#define tree_index int8_t
#define t_float float
#define i8 char
#define children_list int8
#define Score int
#define sIdx int
#define Piece unsigned char
#define int1 int
#define Complex float2
#define sReal16 float4
#define wcl_unsigned_int unsigned int
#define Face int
#define i16 short
#define BSDFEvent int
#define wcl_ulong4 ulong4
#define iscalar_t long
#define pos2 int2
#define cl_float float
#define color4 float4
#define bSize size_t
#define special_func_scalar_type double
#define Bitboard unsigned long long
#define TTScore short
#define hh_float4 float4
#define prob_t float
#define wcl_long long
#define cl_ubyte16 unsigned short
#define realV float
#define boolean char
#define uIntf unsigned int
#define lcg6432_state unsigned long
#define vec3 float3
#define wcl_image2d_t image2d_t
#define cfloat float2
#define opencl_device unsigned
