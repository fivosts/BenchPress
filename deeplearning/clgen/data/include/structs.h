typedef struct InputData {
    int roiY, roiHeight;
    float zeroThreshold, outlierThreshold;
} InputData;

typedef struct OutputData {
    int groupsFinished, dstCount, dstCountZero, dstCountOutlier;
    float srcDstDot[128], dstDstDot[128][128];
} OutputData;

typedef struct {
 float x, y, z;
} Point;

typedef struct node {
  void **pointers;
  int *keys;
  struct node *parent;
  bool is_leaf;
  int num_keys;
  struct node *next;
} node;

typedef struct searchKeyStruct {
  int key;
  __global node *oclNode;
  __global node *nativeNode;
} searchKey;

typedef struct SceneInformation {
  int width;
  int height;
  int pixelCount;
  int groupsSize;
  int bvhSize;
  float proportion_x;
  float proportion_y;
} SceneInformation;

typedef struct {
  int x, y;
  int width, height;
} cairo_rectangle_int_t;

typedef struct positon {
  float x;
  float y;
} position;

typedef struct _NclSymTableListNode {
  struct _NclScopeRec *sr;
  struct _NclSymTableListNode *previous;
} NclSymTableListNode;

typedef struct PixelData {

  uchar3 float3;

  float3 normal;

  float depth;
} PixelData;

typedef struct _PAPhysicsDragData {
  float linearStrength, quadraticStrength, noise;
} PAPhysicsDragData;

typedef struct clrngPhilox432SB_ {
  cl_uint msb, lsb;
} clrngPhilox432SB;

typedef struct clrngPhilox432Counter_ {
  clrngPhilox432SB H, L;
} clrngPhilox432Counter;

typedef struct {
  clrngPhilox432Counter ctr;
  cl_uint deck[4];
  cl_uint deckIndex;
} clrngPhilox432StreamState;

typedef union {
  uint8_t mem_08[4];
  uint16_t mem_16[2];
  uint32_t mem_32[1];
} buffer_32;

typedef struct {
  uint32_t rounds;
  uint32_t length;
  uint32_t final;
  buffer_32 salt[(16 / 4)];
} sha256_salt;

typedef struct {
  float value[64][64];
} BlockMaxima;

typedef struct {
  BlockMaxima blockMaxima[64][64];
} AlignMaxima;

typedef struct {
  AlignMaxima alignMaxima[64][64];
} GlobalMaxima;

typedef struct {
  int noisedepth;
  float bright, contrast;
} BlenderNoiseTexParam;

typedef struct {
  float4 baseFactor;
} ConstantAllocations;

typedef struct {
  float4 m_from;
  float4 m_to;
} b3RayInfo;

typedef struct {

  __constant ConstantAllocations *constantAllocations_min,
      *constantAllocations_max;
  __constant float4 *factors_min, *factors_max;
} ConstantLimits;

typedef struct HSVcolor_struct {
  unsigned char h;
  unsigned char s;
  unsigned char v;
  unsigned char pad;
} HSVcolor;

typedef struct MPI {
  unsigned width;
  unsigned height;
  unsigned interval;
} MPI;

typedef struct SceneGroupStruct {
  int facesSize;
  int facesStart;
  int vertexSize;
} SceneGroupStruct;

typedef struct {
  unsigned int seedBase;
  unsigned int bucketIndex;

  unsigned int filmRegionPixelCount;

} SobolSamplerSharedData;

typedef struct bitcoin_wallet_tmp {
  unsigned long long dgst[8];

} bitcoin_wallet_tmp_t;

typedef struct {
  __local int *int_pointer;
  __local char *char_pointer;
  __local float *float_pointer;
  __local float4 *vector_pointer;
} LocalPrimitivePointerStruct;

typedef struct {
  char data[16];
} DPTYPE_VEC;

typedef struct {
  unsigned char pos;
  unsigned char c[12 + 1];
} ExpandedPattern;

typedef struct ToneMapper {
  int type;
  int applyGamma;
  int shouldClamp;
  float gamma;
  float exposure;
  float key;
  int enableAveraging;
  float averagingAlpha;
} ToneMapper;

typedef struct _cl_mem_ext_host_ptr {

  cl_uint allocation_type;

  cl_uint host_cache_policy;

} cl_mem_ext_host_ptr;

typedef struct _cl_mem_ion_host_ptr {

  cl_mem_ext_host_ptr ext_host_ptr;

  int ion_filedesc;

  void *ion_hostptr;

} cl_mem_ion_host_ptr;

typedef enum atmi_devtype_s {
  ATMI_DEVTYPE_CPU = (1 << 0),
  ATMI_DEVTYPE_iGPU = (1 << 1),
  ATMI_DEVTYPE_dGPU = (1 << 2),
  ATMI_DEVTYPE_GPU = ATMI_DEVTYPE_iGPU | ATMI_DEVTYPE_dGPU,
  ATMI_DEVTYPE_DSP = (1 << 3),
  ATMI_DEVTYPE_ALL = ((1 << 4) - 1)
} atmi_devtype_t;

typedef struct atmi_mem_place_s {

  unsigned int node_id;

  atmi_devtype_t dev_type;

  int dev_id;

  int mem_id;
} atmi_mem_place_t;

typedef struct s_limit {
  int relative;
  float4 high;
  float4 low;
} t_limit;

typedef struct ripemd160_ctx_vector {
  unsigned int h[5];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} ripemd160_ctx_vector_t;

typedef struct _VolumeSlicingConstants {

  int v1[12];
  int v2[12];
  int nSequence[64];
} VolumeSlicingConstants;

typedef struct type_compare {
  int index;
  float value;
} Comparison;

typedef struct {

  unsigned char *_ptr;
  int _cnt;

  unsigned char *_base;
  char _flag;
  char _file;
} FILE;

typedef struct bcrypt_tmp {
  unsigned int E[18];

  unsigned int P[18];

  unsigned int S0[256];
  unsigned int S1[256];
  unsigned int S2[256];
  unsigned int S3[256];

} bcrypt_tmp_t;

typedef struct {
  float16 viewProjMatrix;

  float3 position;

  float3 lookVector;

  float3 ambient;
  float3 diffuse;
  float3 specular;

  float cutOffMin;
  float cutOffMax;

  float exponent;
  float attenuation;

  bool hasShadows;

} cl_spotlight;

typedef struct s_cam {
  float3 pos;
  float3 dir;
  float3 rot;
  float3 updir;
  float3 ldir;
  float3 filter;
  double fov;
  float f_length;
  float aperture;
  float ratio;
  float pr_pl_w;
  float pr_pl_h;
  float dust;
  float brightness;
  float refr_coef;
  int effect;
} t_cam;

typedef struct PV_Stream_ {
  char *name;
  char *mode;
  FILE *_float;
  long filepos;
  long filelength;
  int isfile;
  int verifyWrites;
} PV_Stream;

typedef struct {
  unsigned int cracked;
} out_t;

typedef struct PVHalo_ {
  int lt, rt, dn, up;
} PVHalo;

typedef struct PVLayerLoc_ {
  int nbatch, nx, ny, nf;
  int nbatchGlobal, nxGlobal, nyGlobal;
  int kb0, kx0, ky0;
  PVHalo halo;
} PVLayerLoc;

typedef struct __attribute__((packed)) _svm_ret_type_uint4 {
  uint4 retval;
  uint2 status;
} svm_ret_type_uint4;

typedef struct dpapimk_tmp {

  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[10];
  unsigned int out[10];

  unsigned int userKey[5];

  unsigned long long ipad64[8];
  unsigned long long opad64[8];
  unsigned long long dgst64[16];
  unsigned long long out64[16];

} dpapimk_tmp_t;

typedef struct PVLayerCube_ {
  size_t size;
  int numItems;
  float *data;
  int padding[1];
  PVLayerLoc loc;
  int isSparse;
  long *numActive;
  unsigned int *activeIndices;
} PVLayerCube;

typedef struct {
  char *base;
  size_t stride;
  volatile uint32_t head;
  volatile uint32_t tail;
} clIndexedQueue;

typedef struct PVLayer_ {
  int numNeurons;
  int numExtended;
  int numNeuronsAllBatches;
  int numExtendedAllBatches;

  PV_Stream *activeFP;

  PVLayerLoc loc;
  int xScale, yScale;

  PVLayerCube *activity;
  float *prevActivity;

  float *V;

  void *params;

} PVLayer;

typedef struct streebog512_ctx {
  unsigned long long h[8];
  unsigned long long s[8];
  unsigned long long n[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __constant unsigned long long (*s_sbob_sl64)[256];

} streebog512_ctx_t;

typedef struct streebog512_hmac_ctx {
  streebog512_ctx_t ipad;
  streebog512_ctx_t opad;

} streebog512_hmac_ctx_t;

typedef struct GlobalParameters {
  float planeAmp;
  float sphereAmp;
  float tubeAmp;
  float ringAmp;
  float twistFreqX;
  float twistFreqY;
  float twistOffsetX;
  float twistOffsetY;
  float cameraX;
  float cameraY;
  float cameraZ;
  float dummy1;
  float dummy2;
  float dummy3;
  float dummy4;
  float dummy5;
} GlobalParameters;

typedef struct {
  unsigned int len;
  ushort password[256];
} nt_buffer_t;

typedef struct vfield {
  float probability;
  int state;
} vfield_t;

typedef struct _Rng {
  unsigned int val;
} Rng;

typedef struct _b3MprSimplex_t b3MprSimplex_t;

typedef struct {
  float metric;
  short count;
  unsigned char indexA[3];
  unsigned char indexB[3];
} b2clSimplexCache;

typedef struct b2clMat22 {
  float ex[2];
  float ey[2];
} b2clMat22;

typedef struct b2clFrictionJointData {
  float localAnchorA[2];
  float localAnchorB[2];

  float maxForce;
  float maxTorque;

  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  b2clMat22 linearMass;
  float angularMass;
} b2clFrictionJointData;

typedef struct {

  unsigned char c[12];
} TripcodeKey;

typedef struct {
  unsigned int valueTexIndex, sourceMinTexIndex, sourceMaxTexIndex,
      targetMinTexIndex, targetMaxTexIndex;
} RemapTexParam;

typedef struct {
  int first;
  constant int *firstPtr;
} ConstantPool;

typedef struct Camera {
  float3 pos;
  float3 dir;
  float3 lookAt;
  float3 up;
  float3 right;
  bool from_lookAt;

} Camera;

typedef struct coprthr_cond_attr *coprthr_cond_attr_t;

struct coprthr_cond {};

typedef struct half12 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
} half12;

typedef struct android_backup {
  unsigned int version;
  unsigned int cipher;
  unsigned int iter;
  unsigned int user_salt[16];
  unsigned int ck_salt[16];
  unsigned int user_iv[4];
  unsigned int masterkey_blob[24];

} android_backup_t;

typedef struct b3QuantizedBvhNodeData b3QuantizedBvhNodeData_t;

struct b3QuantizedBvhNodeData {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_escapeIndexOrTriangleIndex;
};

typedef struct {
  uint32_t size[1024];
  uint32_t d[16 * 1024];
} mpzcl32_t;

typedef struct {
  double s_bLow, s_bUp;
  int s_iLow, s_iUp;
} algorithmParams;

typedef struct dpapimk_tmp_v2 {
  unsigned long long ipad64[8];
  unsigned long long opad64[8];
  unsigned long long dgst64[16];
  unsigned long long out64[16];

  unsigned int userKey[8];

} dpapimk_tmp_v2_t;

typedef struct __attribute__((aligned(1))) {
  uchar _type;
} DDFHead;

typedef struct MatrixFloatGlobal {

  __global float *data;
  size_t rowCount;
  size_t columnCount;

} MatrixFloatGlobal;

typedef struct bf {
  unsigned int i;

} bf_t;

typedef struct {
  float x, y;
} Normal;

typedef struct iwork {
  unsigned int iv[4];
  unsigned int data[16];

} iwork_t;

typedef struct float12 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
} float12;

typedef struct {
  int cells[16];
} Mat4X4;

typedef struct ContainsBool {
  bool x;
} ContainsBool;

typedef enum atmi_memtype_s {
  ATMI_MEMTYPE_FINE_GRAINED = 0,
  ATMI_MEMTYPE_COARSE_GRAINED = 1,
  ATMI_MEMTYPE_ANY
} atmi_memtype_t;

/**
 * @brief Task States.
 */
typedef enum atmi_state_s {
  ATMI_UNINITIALIZED = -1,
  ATMI_INITIALIZED = 0,
  ATMI_READY = 1,
  ATMI_DISPATCHED = 2,
  ATMI_EXECUTED = 3,
  ATMI_COMPLETED = 4,
  ATMI_FAILED = 9999
} atmi_state_t;

/**
 * @brief Scheduler Types.
 */
typedef enum atmi_scheduler_s {
  ATMI_SCHED_NONE = 0, // No scheduler, all tasks go to the same queue
  ATMI_SCHED_RR        // Round-robin tasks across queues
} atmi_scheduler_t;

/**
 * @brief ATMI data arg types.
 */
typedef enum atmi_arg_type_s { ATMI_IN, ATMI_OUT, ATMI_IN_OUT } atmi_arg_type_t;

/**
 * @brief ATMI Memory Fences for Tasks.
 */
typedef enum atmi_task_fence_scope_s {
  /**
   * No memory fence applied; external fences have to be applied around the task
   * launch/completion.
   */
  ATMI_FENCE_SCOPE_NONE = 0,
  /**
   * The fence is applied to the device.
   */
  ATMI_FENCE_SCOPE_DEVICE = 1,
  /**
   * The fence is applied to the entire system.
   */
  ATMI_FENCE_SCOPE_SYSTEM = 2
} atmi_task_fence_scope_t;

typedef struct atmi_memory_s {

  unsigned long int capacity;

  atmi_memtype_t type;
} atmi_memory_t;

typedef struct atmi_device_s {

  atmi_devtype_t type;

  unsigned int core_count;

  unsigned int memory_count;

  atmi_memory_t *memories;
} atmi_device_t;

typedef struct BufferVariables bvars;
struct HufmmanVars {
  int read_position;
  int current_read_byte;
  unsigned int CurHuffReadBufPtr;
  unsigned char *CurHuffReadBuf;
};

typedef struct {
  unsigned int state[4];
  unsigned int count[2];
  char buffer[64];
} md5ctx_t;

typedef struct __attribute__((aligned(8))) {
  int globalId;
} clPrintfRestriction;

typedef struct _Triangle {
  float4 a;
  float4 b;
  float4 c;
  float4 a_tc;
  float4 b_tc;
  float4 c_tc;
} Triangle;

typedef struct {

  float charge;

} leaf_moment_t;

typedef struct {
  int a;
  int b;
} mystruct;

typedef struct {
  unsigned int sigmaATexIndex;
} ClearVolumeParam;

typedef struct {
  float invTriangleArea, invMeshArea;

  unsigned int meshIndex, triangleIndex;

  float avarage;
  unsigned int imageMapIndex;
} TriangleLightParam;

typedef struct {
  unsigned short key;
  unsigned int rid;
} row_t;

typedef struct {

  float8 velocity;
  float mass;
  leaf_moment_t moment;

} leaf_value_t;

typedef struct core core;
struct core {
  struct core *next;
  struct core *link;
  short number;
  short accessing_symbol;
  short nitems;
  short items[1];
};

typedef struct {
  unsigned int P[16 + 2];
  unsigned int S[4][256];
} blowfish_context;

typedef struct {
  __global unsigned int *stream_count_kernel__element_counts_min;
  __global unsigned int *stream_count_kernel__element_counts_max;
  __global unsigned int *prefix_scan_kernel__histogram_min;
  __global unsigned int *prefix_scan_kernel__histogram_max;
} WclGlobalLimits;

typedef struct FLOCKParameters {

  float simulation_scale;
  float rest_distance;
  float smoothing_distance;

  float min_dist;
  float search_radius;
  float max_speed;
  float ang_vel;

  float w_sep;
  float w_align;
  float w_coh;
  float w_goal;
  float w_avoid;
  float w_wander;
  float w_leadfoll;

  float slowing_distance;
  int leader_index;

  int num;
  int max_num;
} FLOCKParameters;

typedef struct Vector {
  __global uchar *ptr;
  int offset_first_element_in_bytes;
  int stride_x;
  unsigned int x, y, z;
} Vector;

typedef struct _NclSubscriptSelection {
  long start;
  long finish;
  long stride;
  int is_single;
} NclSubscriptSelection;

typedef struct {
  unsigned int a;
} HitPointGreyTexParam;

typedef struct {
  int x;
  int y;
} StrucTy;

typedef struct {
  float4 v0, v1, v2;
  Vector normal;
  float area;
  float gain_r, gain_g, gain_b;
} TriangleLight;

typedef struct {
  unsigned int saltlen;
  uchar salt[8];
  uchar prefix;
} crypt_md5_salt;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_Rs __attribute__((aligned(4)));
  unsigned int idx_Rd;
} AshikhminDElem;

typedef struct {
  float value[16][16];
} ScoringsMatrix;

typedef struct _pass_settings_t {
  uchar max_diff_depth, max_glossy_depth, max_refr_depth, max_transp_depth,
      max_total_depth;
  uchar termination_start_depth;
  uchar pad[2];
  unsigned int flags;
} pass_settings_t;

typedef struct salt {
  unsigned int salt_buf[64];
  unsigned int salt_buf_pc[64];

  unsigned int salt_len;
  unsigned int salt_len_pc;
  unsigned int salt_iter;
  unsigned int salt_iter2;
  unsigned int salt_sign[2];

  unsigned int digests_cnt;
  unsigned int digests_done;

  unsigned int digests_offset;

  unsigned int scrypt_N;
  unsigned int scrypt_r;
  unsigned int scrypt_p;

} salt_t;

typedef struct dim_str {

  int cur_arg;
  int arch_arg;
  int cores_arg;
  int boxes1d_arg;

  long number_boxes;
  long box_mem;
  long space_elem;
  long space_mem;
  long space_mem2;

} dim_str;

typedef struct {
  salt_t pbkdf2;
  uint8_t header[96];
} diskcryptor_salt_t;

typedef struct {
  ScoringsMatrix metaMatrix[64][64];
} GlobalMatrix;

typedef struct {
  float rgb_scale_error;
  float rgb_luma_error;
  float luminance_error;
  float alpha_drop_error;
  float rgb_drop_error;
  int can_offset_encode;
  int can_blue_contract;
} encoding_choice_errors;

typedef struct {

  volatile uint32_t next;
} clIndexedQueue_item;

typedef struct _NclGenProcFuncInfo {
  int nargs;
  struct _NclArgTemplate *theargs;
  struct _NclSymbol *thesym;
  struct _NclScopeRec *thescope;
} NclGenProcFuncInfo;

typedef struct {
  unsigned char m_code_size[128];
  signed short m_look_up[256], m_tree[512];
} tinfl_huff_table;

typedef struct {
  unsigned int d[((256 / 8) / sizeof(unsigned int))];
} bignum;

typedef struct TF_TString_Large {
  size_t size;
  size_t cap;
  char *ptr;
} TF_TString_Large;

typedef struct {
  char data[32];
} channel_sign_scal;

typedef struct T {
  int *a[2];
} T;

typedef struct {
  float r, g, b;
} Spectrum;

typedef struct {
  float r, g, b;
} Pixel;

typedef struct {
  Spectrum gain;
  float thetaS;
  float phiS;
  float zenith_Y, zenith_x, zenith_y;
  float perez_Y[6], perez_x[6], perez_y[6];
} SkyLight;

typedef struct b2clPulleyJointData {
  float groundAnchorA[2];
  float groundAnchorB[2];
  float lengthA;
  float lengthB;
  float localAnchorA[2];
  float localAnchorB[2];
  float pulleyConstant;
  float ratio;
  float uA[2];
  float uB[2];
  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  float mass;
} b2clPulleyJointData;

typedef struct _neural_memory_tag {
  unsigned int format;
  unsigned int feature_offset;
  unsigned int spatial_offset;
  unsigned int vector_size;
  unsigned int data_offset;
  unsigned int data[1];
} neural_memory;

typedef struct knode {
  int location;
  int indices[256 + 1];
  int keys[256 + 1];
  bool is_leaf;
  int num_keys;
} knode;

typedef struct Keypoint {
  int x;
  int y;
  float strength;
  float scale;
  float orientation;
  int tracking_status;
  float error;
} Keypoint;

typedef struct __attribute__((packed)) _svm_ret_type_uint2 {
  uint2 retval;
  uint2 status;
} svm_ret_type_uint2;

typedef struct s_body {
  cl_float4 position;
  cl_float4 velocity;
} t_body;

typedef struct s_workunit {
  int id;
  int localcount;
  int npadding;
  int neighborcount;
  int mpadding;
  cl_float4 *N;
  cl_float4 *M;
  cl_float4 *V;
  t_body *local_bodies;
  char is_last;
} t_workunit;

typedef struct dpapimk {
  unsigned int version;
  unsigned int context;

  unsigned int SID[32];
  unsigned int SID_len;
  unsigned int SID_offset;

  unsigned int iv[4];
  unsigned int contents_len;
  unsigned int contents[128];

} dpapimk_t;

typedef struct tc_tmp {
  unsigned int ipad[16];
  unsigned int opad[16];

  unsigned int dgst[64];
  unsigned int out[64];

} tc_tmp_t;

typedef struct element {
  uint32_t digest_index;
  uint32_t parent_bucket_data;

} element_t;

typedef struct src_local_bucket {
  element_t data[17];
} src_local_bucket_t;

typedef struct {
  const uchar v[256];
} odf_password;

typedef struct {
  uchar state;
} Cell;

typedef struct VolumeParameters_t {
  float16 modelToWorld;
  float16 worldToModel;
  float16 worldToTexture;
  float16 textureToWorld;
  float16 textureToIndex;
  float16 indexToTexture;
  float16 textureSpaceGradientSpacing;
  float3 worldSpaceGradientSpacing;
  float formatScaling;
  float formatOffset;
  float signedFormatScaling;
  float signedFormatOffset;
  char padding__[32];
} VolumeParameters;

typedef struct sha256crypt_tmp {

  unsigned int alt_result[8];
  unsigned int p_bytes[64];
  unsigned int s_bytes[64];

} sha256crypt_tmp_t;

typedef struct {

  float8 force;

} force_t;

typedef struct clb2Contact {
  int float3;
  int indexA;
  int indexB;
  float friction;
  float2 normal;
  float invMassA;
  float invIA;
  float invMassB;
  float invIB;
} clb2Contact;

typedef struct __attribute__((aligned(4))) {
  uchar texType;
  unsigned int width __attribute__((aligned(4))), height;
  int offsetData;
} NormalMapTexture;

typedef struct half1 {
  float s0;
} half1;

typedef struct {
  unsigned int w, jsr, jcong;
} kiss99_t;

typedef struct {
  unsigned int matAIndex, matBIndex;
  unsigned int mixFactorTexIndex;
} MixParam;

typedef struct {
  Spectrum float3;
  int useVisibilityMapCache;
} ConstantInfiniteLightParam;

typedef struct sha384_ctx {
  unsigned long long h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];
  unsigned int w4[4];
  unsigned int w5[4];
  unsigned int w6[4];
  unsigned int w7[4];

  int len;

} sha384_ctx_t;

typedef struct {
  float r, g, b;
} MatteParam;

typedef struct {
  float scale_;
  float offset_;
} RealWorldMapping;

typedef struct gz_header_s {
  int text;
  ulong time;
  int xflags;
  int os;
  unsigned char *extra;
  unsigned int extra_len;
  unsigned int extra_max;
  unsigned char *name;
  unsigned int name_max;
  unsigned char *comment;
  unsigned int comm_max;
  int hcrc;
  int done;

} gz_header;

typedef struct {
  int3 start_index;
  int3 end_index;
  float3 start_continuous_index;
  float3 end_continuous_index;
} GPUImageFunction3D;

typedef struct {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_escapeIndexOrTriangleIndex;
} btQuantizedBvhNode;

typedef struct _VSOutput {
  float4 position;
  float4 float3;
  float4 texcoord0;
} VSOutput;

typedef struct _NclFileInfo {
  char filename[256];
  int level;
  unsigned int offset;
  struct _NclScopeRec *filescope;
} NclFileInfo;

typedef struct {
  float r, g, b;
  float exponent;
  int specularBounce;
} MetalParam;

typedef struct {
  MatteParam matte;
  MetalParam metal;
  float matteFilter, totFilter, mattePdf, metalPdf;
} MatteMetalParam;

typedef struct __attribute__((aligned(4))) Node {
  int4 n __attribute__((aligned(4)));
} Node;

typedef struct differential {
  float dx;
  float dy;
} differential;

typedef struct {
  float3 position, next_velocity;
  int collision_happened;
  float time_elapsed;
} collision_response;

typedef struct {
  double lo, hi;
} v2double;

typedef struct clb2PositionSolverManifold {
  float2 normal;
  float2 point;
  float separation;
} clb2PositionSolverManifold;

typedef struct __attribute__((aligned(1))) {
  uchar fresnelType;
} FresnelHead;

typedef struct {
  float dist;
  unsigned int id;
} dist_id;

typedef struct {
  uint16 s0, s1, s2, s3;
} v4uint16;

typedef struct {
  int int_value;
  char char_value;
  float float_value;
  float4 vector_value;
} PrimitiveStruct;

typedef struct jks_sha1 {
  unsigned int checksum[5];
  unsigned int iv[5];
  unsigned int enc_key_buf[4096];
  unsigned int enc_key_len;
  unsigned int der[5];
  unsigned int alias[16];

} jks_sha1_t;

typedef struct wpa_pmk_tmp {
  unsigned int out[8];

} wpa_pmk_tmp_t;

typedef struct __attribute__((aligned(4))) {
  FresnelHead head;
  float etaExt __attribute__((aligned(4))), etaInt;
} FresnelDielectric;

typedef struct Mat33 {
  float data[3][3];
} Mat33;

typedef struct md5_ctx {
  unsigned int h[4];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} md5_ctx_t;

typedef struct md5_hmac_ctx {
  md5_ctx_t ipad;
  md5_ctx_t opad;

} md5_hmac_ctx_t;

typedef struct {
  int data[6];
} INT6;

typedef struct _param_type {
  char atomtype[512];
  int ntypes, axis;
  double charge, epsilon, sigma, omega, dr, pol, c6, c8, c10;
  double last_charge, last_epsilon, last_sigma, last_dr, last_omega, last_pol,
      last_c6, last_c8, last_c10;
  struct _param_type *next;
} param_t;

typedef struct _hit_data_t {
  int mask, obj_index, prim_index;
  float t, u, v;
  float2 ray_id;
} hit_data_t;

typedef struct ctx_t {

  uint8_t b[128];

  cl_ulong h[8];

  cl_ulong t[2];

  cl_uint c;
} ctx_t;

typedef struct {
  float phi1;
  float phi2;
  float p1;
  float p2;
} pstate;

typedef struct {
  unsigned int index;
  uint4 v[4];
} dag_gen_share;

typedef struct {
  float dx;
  float dy;
  float dz;

  int nx;
  int ny;
  int nz;

  float x0;
  float y0;

  float sep;

  float x0_temp_det;
  float y0_temp_det;
  unsigned int x_temp_numdets;
  unsigned int y_temp_numdets;
  float x_temp_sepdets;
  float y_temp_sepdets;
  float temp_det_r;
  unsigned int temp_rort;
  unsigned long temp_bins;
  unsigned long max_temp;
} DetStruct;

typedef struct _pair {
  int frozen;
  int rd_excluded, es_excluded;
  int attractive_only;
  int recalculate_energy;
  double lrc;
  double last_volume;
  double epsilon, sigma;
  double r, rimg, dimg[3];
  double d_prev[3];
  double rd_energy, es_real_energy, es_self_intra_energy;
  double sigrep;
  double c6, c8, c10;
  struct _atom *atom;
  struct _molecule *molecule;
  struct _pair *next;
} pair_t;

typedef struct luks_tmp {
  unsigned int ipad32[8];
  unsigned long long ipad64[8];

  unsigned int opad32[8];
  unsigned long long opad64[8];

  unsigned int dgst32[32];
  unsigned long long dgst64[16];

  unsigned int out32[32];
  unsigned long long out64[16];

} luks_tmp_t;

typedef struct {
  char data[16];
} lane_data;

typedef struct {
  unsigned long long hash;
  unsigned int bestmove;
  short score;
  unsigned char flag;
  unsigned char depth;
} TTE;

typedef struct {
  lane_data lane[32];
} channel_vec;

typedef struct {
  channel_vec w_vec[6];
} weight_tmp;

typedef struct {
  uint3 s0, s1, s2, s3;
} v4uint3;

typedef struct _mesh_instance_t {
  float bbox_min[3];
  unsigned int tr_index;
  float bbox_max[3];
  unsigned int mesh_index;
} mesh_instance_t;

typedef struct CopyBufFuncs {
  char read[64][64];
  char write[64];
  char readGeneric[64][64];
  char writeGeneric[64];
} CopyBufFuncs;

typedef struct {
  char data[6 / 2 - 1];
} w_vec_pool;

typedef struct {
  int field;
} CPTypedef;

typedef struct {
  int m_seed;
  float m_frequency;
  float m_perturbFrequency;
  float m_cellularJitter;
  int m_cellularDistanceIndex0;
  int m_cellularDistanceIndex1;
  int m_smoothing;
  int m_noiseType;

  int m_octaves;
  float m_lacunarity;
  float m_gain;
  int m_fractalType;

  float m_fractalBounding;

  int m_cellularDistanceFunction;
  int m_cellularReturnType;

  float m_perturbSeed;
  float m_perturbAmp;
  int m_perturb;
  int m_perturbOctaves;
  float m_perturbGain;
  float m_perturbLacunarity;

  float m_perturbBounding;
  int m_perturbSmoothing;
} Snapshot;

typedef struct {
  unsigned int s1, s2, s3;
} Seed;

typedef struct {
  float3 v0;
  float3 v1;
  float3 v2;
} TriangleVerts;

typedef struct {

  unsigned char c[12];
} Tripcode;

typedef struct {
  unsigned int s0, s1, s2, s3;
} v4uint;

typedef struct Info {
  int dims;
  int offset;
  int sizes[5];
  int strides[5];
} Info;

typedef struct {
  float x;
  float y;
} floatk3;

typedef struct {
  float3 verts[3];
} Polygon;

typedef struct pkzip_extra {
  unsigned int buf[2];
  unsigned int len;

} pkzip_extra_t;

typedef struct securezip {
  unsigned int data[36];
  unsigned int file[16];
  unsigned int iv[4];
  unsigned int iv_len;

} securezip_t;

typedef struct {
  float4 albedo;
  float r0;
  float smoothness;
  float2 dummy;
} TMaterial;

typedef struct {
  int type;
  int materialID;
} CommonObject;

typedef struct {
  float3 position;
  float3 rotation;
  float3 velocity_linear;
  float3 velocity_angular;
} RigidBodyMotion;

typedef struct {
  uint32_t cracked;
} strip_out;

typedef struct {
  unsigned int avg_y;

  unsigned int avg_r;
  unsigned int avg_gr;
  unsigned int avg_gb;
  unsigned int avg_b;
  unsigned int valid_wb_count;

  unsigned int f_value1;
  unsigned int f_value2;
} XCamGridStat;

typedef struct _f_point {
  float x;
  float y;
} f_point_t;

typedef struct {
  int ReadPos;
  float PastInputs[8];
  float PastOutputs[8];
} DiffEqn;

typedef struct _f_subsampling_info {
  f_point_t point;
  float xbin;
  float ybin;
  float val;
} f_ss_kinfo;

typedef struct office2013_tmp {
  unsigned long long out[8];

} office2013_tmp_t;

typedef struct {
  double x, y;
} TVector;

typedef struct {
  int field;
} RTypedef;

typedef struct {
  RTypedef rstruct;
} PRTypedef;

typedef struct VglClShape {
  int ndim;
  int shape[(10 + 1)];
  int offset[(10 + 1)];
  int size;
} VglClShape;

typedef struct _NclVarInfo {
  char varname[256];
  int level;
  int datatype;
  unsigned int offset;
} NclVarInfo;

typedef struct TINYMT64J_T {
  cl_ulong s0;
  cl_ulong s1;
} tinymt64j_t;

typedef struct {
  float3 ambient;
  float3 diffuse;
  float3 specular;
  float shininess;
} cl_material;

typedef struct _CombinerShader {

} CombinerShader;

typedef struct double_nested_struct {
  int x;
  struct double_nested {
    struct inner_inner {
      char y;
      int q;
    } inner_inner;
  } inner;

  short w;
} double_nested_struct;

typedef struct struct_arr16 {
  int arr[16];
} struct_arr16;

typedef struct sha384_ctx_vector {
  unsigned long long h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];
  unsigned int w4[4];
  unsigned int w5[4];
  unsigned int w6[4];
  unsigned int w7[4];

  int len;

} sha384_ctx_vector_t;

typedef struct wpa_pbkdf2_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} wpa_pbkdf2_tmp_t;

typedef struct {
  int data[6];
} w_vec_data_32;

typedef struct CvHuMoments {
  double hu1, hu2, hu3, hu4, hu5, hu6, hu7;
} CvHuMoments;

typedef struct {
  int dummy;
} used_params_struct;

typedef struct {
  unsigned int length;
  uchar v[255];
} crypt_md5_password;

typedef struct s_disk {
  float3 pos;
  float r;
  int related;
} t_disk;

typedef struct {
  int type;
  float param1;
  float param2;
  int param3;
} PostProcessingInfo;

typedef struct b2clMouseJointData {
  float localAnchorB[2];
  float targetA[2];
  float frequencyHz;
  float dampingRatio;
  float beta;

  float maxForce;
  float gamma;

  float rB[2];
  float localCenterB[2];
  float invMassB;
  float invIB;
  b2clMat22 mass;
  float C[2];
} b2clMouseJointData;

typedef struct {
  int width;
  int height;
  int quality;
  float fov;
  float intensity;
} sParamsSSAO;

typedef struct {
  float2 m_buffer[2];
  float2 m_vertices[8];
  int m_count;
  float m_radius;
} b2clDistanceProxy;

typedef struct {
  float tmat[6];
} tmat_t;

typedef struct TfLiteIntArray {
  int size;

  int data[];

} TfLiteIntArray;

typedef struct b2clTransform {
  float2 p;
  float2 q;
} b2clTransform;

typedef struct {
  float2 vertices[8];
  float2 normals[8];
  int count;
} b2clTempPolygon;

typedef struct {
  salt_t pbkdf2;
  uint8_t encseed[1024];
  uint32_t eslen;
} ethereum_salt_t;

typedef struct {
  cl_float3 direction;
  float coefficient;
} _Speaker_unalign;

typedef struct {
  b2clDistanceProxy proxyA;
  b2clDistanceProxy proxyB;
  b2clTransform transformA;
  b2clTransform transformB;
  bool useRadii;
} b2clDistanceInput;

typedef struct tag_iSize {
  int cols;
  int rows;
  int step;
  int size;
} Size2d;

typedef struct {
  int num_texels;
  int num_weights;
  uint8_t texel_num_weights[216];
  uint8_t texel_weights_int[216][4];
  float texel_weights_float[216][4];
  uint8_t texel_weights[216][4];
  uint8_t weight_num_texels[64];
  uint8_t weight_texel[64][216];
  uint8_t weights_int[64][216];
  float weights_flt[64][216];
} decimation_table;

typedef struct {

  float8 position;
  leaf_value_t value;

} leaf_t;

typedef struct {
  int sum;
  unsigned int sse;
} SUM_SSE;

typedef struct {
  int8 s0, s1, s2, s3;
} v4int8;

typedef struct bitcoin_wallet {
  unsigned int cry_master_buf[64];
  unsigned int ckey_buf[64];
  unsigned int public_key_buf[64];

  unsigned int cry_master_len;
  unsigned int ckey_len;
  unsigned int public_key_len;

} bitcoin_wallet_t;

typedef struct KernelCurves {

  float encasing_ratio;
  int curveflags;
  int subdivisions;
  int pad1;

  float minimum_width;
  float maximum_width;
  float curve_epsilon;
  int pad2;
} KernelCurves;

typedef struct {
  float x;
  float y;
  float r;

  float mutr;
  float mua;
  float g;
  float n;

  float flc;
  float muaf;
  float eY;
  float albedof;
} IncStruct;

typedef struct _platform_info {
  cl_platform_id id;
  char *profile;
  char *version;
  char *name;
  char *vendor;
  char *extensions;
} _PLATFORM_INFO;

typedef struct {
  Spectrum radiance;

  float totalI;

  unsigned int largeMutationCount, smallMutationCount;
  unsigned int current, proposed, consecutiveRejects;

  float weight;
  Spectrum currentRadiance;
} MetropolisSampleWithoutAlphaChannel;

typedef struct {
  unsigned int mt[624];
  int mti;
} mt19937_state;

typedef struct {
  float imag;
} cplx_t;

typedef struct {
  int n, lda, j0;
  short ipiv[32];
} slaswp_params_t;

typedef struct {
  float4 m_linear;
  float4 m_worldPos[4];
  float4 m_center;
  float m_jacCoeffInv[4];
  float m_b[4];
  float m_appliedRambdaDt[4];

  float m_fJacCoeffInv[2];
  float m_fAppliedRambdaDt[2];

  unsigned int m_bodyA;
  unsigned int m_bodyB;

  int m_batchIdx;
  unsigned int m_paddings[1];
} Constraint4;

typedef struct {
  float2 pos;
  float radius;
} Circle;

typedef struct IsectPrecalc {

  int kx, ky, kz;

  float Sx, Sy, Sz;
} IsectPrecalc;

typedef struct {
  float3 c0;
  float3 c1;
} aabb;

typedef struct {
  float dx;
} state;

typedef struct {
  float mass;
  float posx;
  float posy;
  float posz;
  float velx;
  float vely;
  float velz;
  float accx;
  float accy;
  float accz;
  float dsq;
} OctTreeLeafNode;

typedef struct inter_index_param {
  int pred_mode;
  int buf_offset;
  int dst_offset;
  int dst_stride;
  int pre_stride;
  int src_num;
  int w;
  int h;
  int sub_x;
  int sub_y;
} INTER_INDEX_PARAM_GPU;

typedef struct _InputMapData {
  union {
    struct {
      float3 value;
    } float_value;
    struct {
      int idx;
      int placeholder[2];
      int type;
    } int_values;
  };
} InputMapData;

typedef struct {
  int nthreads;
  long n_isotopes;
  long n_gridpoints;
  int lookups;
  int HM;
  int grid_type;
  int hash_bins;
  int particles;
  int simulation_method;
  int binary_mode;
  int kernel_id;
} Inputs;

typedef struct _vx_meta_format *vx_meta_format;

enum vx_type_e {
  VX_TYPE_INVALID = 0x000,
  VX_TYPE_CHAR = 0x001,
  VX_TYPE_INT8 = 0x002,
  VX_TYPE_UINT8 = 0x003,
  VX_TYPE_INT16 = 0x004,
  VX_TYPE_UINT16 = 0x005,
  VX_TYPE_INT32 = 0x006,
  VX_TYPE_UINT32 = 0x007,
  VX_TYPE_INT64 = 0x008,
  VX_TYPE_UINT64 = 0x009,
  VX_TYPE_FLOAT32 = 0x00A,
  VX_TYPE_FLOAT64 = 0x00B,
  VX_TYPE_ENUM = 0x00C,
  VX_TYPE_SIZE = 0x00D,
  VX_TYPE_DF_IMAGE = 0x00E,

  VX_TYPE_BOOL = 0x010,

  VX_TYPE_RECTANGLE = 0x020,
  VX_TYPE_KEYPOINT = 0x021,
  VX_TYPE_COORDINATES2D = 0x022,
  VX_TYPE_COORDINATES3D = 0x023,
  VX_TYPE_COORDINATES2DF = 0x024,
  VX_TYPE_HOG_PARAMS = 0x028,
  VX_TYPE_HOUGH_LINES_PARAMS = 0x029,
  VX_TYPE_LINE_2D = 0x02A,
  VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS = 0x02B,

  VX_TYPE_USER_STRUCT_START = 0x100,
  VX_TYPE_VENDOR_STRUCT_START = 0x400,
  VX_TYPE_KHRONOS_OBJECT_START = 0x800,
  VX_TYPE_VENDOR_OBJECT_START = 0xC00,

  VX_TYPE_KHRONOS_STRUCT_MAX = VX_TYPE_USER_STRUCT_START - 1,

  VX_TYPE_USER_STRUCT_END = VX_TYPE_VENDOR_STRUCT_START - 1,
  VX_TYPE_VENDOR_STRUCT_END = VX_TYPE_KHRONOS_OBJECT_START - 1,
  VX_TYPE_KHRONOS_OBJECT_END = VX_TYPE_VENDOR_OBJECT_START - 1,
  VX_TYPE_VENDOR_OBJECT_END = 0xFFF,

  VX_TYPE_REFERENCE = 0x800,
  VX_TYPE_CONTEXT = 0x801,
  VX_TYPE_GRAPH = 0x802,
  VX_TYPE_NODE = 0x803,
  VX_TYPE_KERNEL = 0x804,
  VX_TYPE_PARAMETER = 0x805,
  VX_TYPE_DELAY = 0x806,
  VX_TYPE_LUT = 0x807,
  VX_TYPE_DISTRIBUTION = 0x808,
  VX_TYPE_PYRAMID = 0x809,
  VX_TYPE_THRESHOLD = 0x80A,
  VX_TYPE_MATRIX = 0x80B,
  VX_TYPE_CONVOLUTION = 0x80C,
  VX_TYPE_SCALAR = 0x80D,
  VX_TYPE_ARRAY = 0x80E,
  VX_TYPE_IMAGE = 0x80F,
  VX_TYPE_REMAP = 0x810,
  VX_TYPE_ERROR = 0x811,
  VX_TYPE_META_FORMAT = 0x812,
  VX_TYPE_OBJECT_ARRAY = 0x813,

  VX_TYPE_TENSOR = 0x815,

};

typedef struct edge {
  ulong from;
  ulong to;
  ulong cost;
} edge;

typedef struct SeamNodeInfo_s {
  int4 localspaceMin;
  float4 position;
  float4 normal;
} SeamNodeInfo;

typedef struct {
  float x, y, z;

} THREE_VECTOR;

typedef struct {
  lane_data wvec[6];
} data_wng;

typedef struct {
  long quot, rem;
} ldiv_t;

typedef struct {
  int lower;
  int upper;
  bool execute;
  bool last;
} nanos_ws_item_loop_t;

typedef struct _atom {
  int id, bond_id;
  char atomtype[512];
  int frozen, adiabatic, spectre, target;
  double mass, charge, polarizability, epsilon, sigma, omega;
  double c6, c8, c10, c9;
  double es_self_point_energy;
  double pos[3], wrapped_pos[3];
  double ef_static[3], ef_static_self[3], ef_induced[3], ef_induced_change[3];
  double mu[3], old_mu[3], new_mu[3];
  double dipole_rrms;
  double rank_metric;
  int gwp_spin;
  double gwp_alpha;
  int site_neighbor_id;
  pair_t *pairs;
  double lrc_self, last_volume;
  struct _atom *next;

} atom_t;

typedef struct {
  uint32_t sk[3 * 32];
} des3_context;

typedef struct ripemd160_ctx {
  unsigned int h[5];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} ripemd160_ctx_t;

typedef struct TINYMT64WP_T {
  cl_ulong s0;
  cl_ulong s1;
  cl_uint mat1;
  cl_uint mat2;
  cl_ulong tmat;
} tinymt64wp_t;

typedef struct _molecule {
  int id;
  char moleculetype[512];
  double mass;
  int frozen, adiabatic, spectre, target;
  double com[3], wrapped_com[3];
  double iCOM[3];
  int nuclear_spin;
  double rot_partfunc_g, rot_partfunc_u, rot_partfunc;
  atom_t *atoms;
  struct _molecule *next;
} molecule_t;

typedef struct SubproblemDim {
  size_t x;
  size_t y;

  size_t bwidth;
  size_t itemX;

  size_t itemY;

} SubproblemDim;

typedef struct PGranularity {

  unsigned int wgSize[2];

  unsigned int wgDim;

  unsigned int wfSize;

  unsigned int numWGSpawned[2];

  unsigned int maxWorkGroupSize;
} PGranularity;

typedef struct {
  SubproblemDim subdims[4];
  PGranularity pgran;
} DecompositionStruct;

typedef struct {
  unsigned int sequence;
  unsigned int target;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int valueX;
  unsigned int valueY;
  float score;
  float maxScore;
  float posScore;
} StartingPoint;

typedef struct _device_info {
  cl_device_id id;
  cl_device_type type;
  cl_uint max_compute_units;
  cl_uint max_work_item_dimensions;
  size_t *max_work_item_sizes;
  size_t max_work_group_size;

  cl_uint max_clock_frequency;
  cl_uint address_bits;

  cl_ulong max_mem_alloc_size;
  cl_bool image_support;
  cl_uint max_samplers;
  size_t max_parameter_size;
  cl_device_fp_config single_fp_config;
  cl_device_mem_cache_type mem_cache_type;

  cl_ulong global_mem_cacheline_size;
  cl_ulong global_mem_cache_size;
  cl_ulong global_mem_size;
  cl_ulong max_constant_buffer_size;
  cl_uint max_constant_args;

  cl_device_local_mem_type local_mem_type;
  cl_ulong local_mem_size;
  cl_bool error_correction;

  size_t profiling_timer_resolution;
  cl_bool endian_little;
  cl_bool available;

  cl_bool compiler_available;
  cl_device_exec_capabilities exec_capabilities;
  cl_command_queue_properties queue_properties;

  char *name;
  char *vendor;
  char *driver_version;
  char *profile;
  char *version;
  char *extensions;
} _DEVICE_INFO;

typedef struct {
  Vector absoluteLightPos, absoluteLightDir;
  Spectrum emittedFactor;
  float radius;
} LaserLightParam;

typedef struct NhlGenArrayRec_ *NhlGenArray;

typedef enum _NhlErrType {
  NhlFATAL = -4,
  NhlWARNING = -3,
  NhlINFO = -2,
  NhlNOERROR = -1
} NhlErrorTypes;

typedef struct {
  uint2 s0, s1, s2, s3;
} v4uint2;

typedef struct _cl_spacetime_grid_t {
  double m_mass[((32) * (32) * (32))];
  double m_pull[((32) * (32) * (32))];
  double4 m_position[((32) * (32) * (32))];
  double4 m_budge[((32) * (32) * (32))];
} cl_spacetime_grid_t;

typedef struct {
  StartingPoint startingPoint[(64 * 2 * 1000)];
} StartingPoints;

typedef struct {
  uint32_t head;
  uint32_t free;
} clqueue_32;

typedef struct pbkdf1_sha1_tmp {

  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int out[5];

} pbkdf1_sha1_tmp_t;

typedef struct {
  double2 lo, hi;
} v2double2;

typedef struct {
  float M[3][2];
} ago_affine_matrix_t;

typedef struct {
  float4 l0;
  float4 l1;
  float4 l2;
  float4 l3;
} Mueller;

typedef struct {
  float2 center;
  float radius;
} target_area_t;

typedef struct {
  void *pfn_notify;
  void *user_data;
} fp_ud_t;

typedef struct {
  unsigned char value[2][2];
} Direction;

typedef struct DeviceHwInfo {
  unsigned int wavefront;
  unsigned int channelSize;
  unsigned int bankSize;
  unsigned int l1CacheAssoc;
} DeviceHwInfo;

typedef struct _NclSymbol {
  int type;
  int ind;
  int level;
  char name[256];
  unsigned int offset;
  union {
    struct _NclVarInfo *var;
    struct _NclFileInfo *file;
    struct _NclFileVarInfo *fvar;
    struct _NclVisBlkInfo *visblk;
    struct _NclProcFuncInfo *procfunc;
    struct _NclBuiltInFuncInfo *bfunc;
    struct _NclBuiltInProcInfo *bproc;
    struct _NclSharedLibraryInfo *package;
    struct _NhlClassRec *obj_class_ptr;
  } u;
  struct _NclSymbol *symnext;
  struct _NclSymbol *sympre;
} NclSymbol;

typedef struct half14 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
  float se;
} half14;

typedef struct _ByteRange {
  uint offset;
  uint length;
} ByteRange;

typedef struct {
  cl_float3 reflection;
  cl_float3 float3;
  float power;
} Material;

typedef struct {
  float4 pmin;
  float4 pmax;
} bbox;

typedef struct struct_padding_arg {
  char i1;
  long f;
} struct_padding_arg;

typedef struct {
  int rect_idx;
  int lbpmap[8];
  float pos;
  float neg;
} weak_classifier;

typedef struct _QueueElement {
  int blockx, blocky, blockz;
  int state;
} QueueElement;

typedef struct {
  float startX;
  float offsetX;
  int lengthX;
  float startY;
  float offsetY;
  int lengthY;
  float startZ;
  float offsetZ;
  int lengthZ;
} ImplicitCube;

typedef struct {
  w_vec_data_32 lane[32];
} channel_wvec_32_wng;

typedef struct md4_ctx {
  unsigned int h[4];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} md4_ctx_t;

typedef struct {
  unsigned int key;
  unsigned long long val;

} hcstat_table_t;

typedef struct {
  float2 a, b;
} ComplexPair;

typedef struct triangle_object_data {
  float3 p1;
  float3 p2;
  float3 p3;
  float3 n1;
  float3 n2;
  float3 n3;
} triangle_object_data;

typedef struct {
  int localid;
  int type;
  Vector position;
  Vector velocity;
  float radius;
  int colour;
} agentinfo;

typedef struct {
  float4 m_childPosition;
  float4 m_childOrientation;
  int m_shapeIndex;
  int m_unused0;
  int m_unused1;
  int m_unused2;
} btGpuChildShape;

typedef struct {
  long long quot, rem;
} lldiv_t;

typedef struct sha224_ctx {
  unsigned int h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha224_ctx_t;

typedef struct office2010_tmp {
  unsigned int out[5];

} office2010_tmp_t;

typedef struct sha224_hmac_ctx {
  sha224_ctx_t ipad;
  sha224_ctx_t opad;

} sha224_hmac_ctx_t;

typedef struct devise_hash {
  unsigned int salt_buf[64];
  int salt_len;

  unsigned int site_key_buf[64];
  int site_key_len;

} devise_hash_t;

typedef struct {
  uchar v[15];
  uchar length;
} phpass_password;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_T __attribute__((aligned(4)));
  float etaExt;
  float etaInt;
  unsigned int idx_Fresnel;
} SpecularTElem;

typedef struct {
  float s00;
  float s01;
  float s11;
} Matrix2x2ConjSymmetric;

typedef struct _slab_header {
  unsigned int offset;
  unsigned int outindex;
  unsigned int outspan;
} slab_header;

typedef struct cloudkey {
  unsigned int data_len;
  unsigned int data_buf[512];

} cloudkey_t;

typedef struct {
  cl_float3 pos;
  cl_float3 float3;
} Light;

typedef struct KernelIntegrator {

  int use_direct_light;
  int use_ambient_occlusion;
  int num_distribution;
  int num_all_lights;
  float pdf_triangles;
  float pdf_lights;
  float inv_pdf_lights;
  int pdf_background_res;

  int min_bounce;
  int max_bounce;

  int max_diffuse_bounce;
  int max_glossy_bounce;
  int max_transmission_bounce;

  int transparent_min_bounce;
  int transparent_max_bounce;
  int transparent_shadows;

  int no_caustics;
  float filter_glossy;

  int seed;

  int layer_flag;

  float sample_clamp;

  int branched;
  int aa_samples;
  int diffuse_samples;
  int glossy_samples;
  int transmission_samples;
  int ao_samples;
  int mesh_light_samples;
  int subsurface_samples;

  int use_lamp_mis;

  int sampling_pattern;

  int pad;
} KernelIntegrator;

typedef struct MQEncoder {
  short L;

  unsigned short A;
  unsigned int C;
  unsigned char CT;
  unsigned char T;

  __global unsigned char *outbuf;

  int CXMPS;
  unsigned char CX;

  unsigned int Ib0;
  unsigned int Ib1;
  unsigned int Ib2;
  unsigned int Ib3;
  unsigned int Ib4;
  unsigned int Ib5;
} MQEncoder;

typedef struct {
  float4 Direction;
} ParticleForce;

typedef struct {
  short data[6];
} w_vec_data_16;

typedef struct SubgVarNames {

  const char *subgCoord;
  const char *itemId;
} SubgVarNames;

typedef struct myUnpackedStruct {
  char c;
  char2 vec;
} testStruct;

typedef struct {
  w_vec_data_16 vec[16];
} data_16_wng;

typedef struct {
  unsigned int bucketIndex;
} RandomSamplerSharedData;

typedef struct VglClStrEl {
  float data[256];
  int ndim;
  int shape[(10 + 1)];
  int offset[(10 + 1)];
  int size;
} VglClStrEl;

typedef struct {
  float s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14,
      s15, s16, s17, s18, s19, s20, s21, s22, s23, s24;
  float carry;
  float dummy;
  int in24;
  int stepnr;
} ranluxcl_state_t;

typedef struct {
  Tripcode tripcode;
  TripcodeKey key;
} TripcodeKeyPair;

typedef struct Image {
  __global uchar *ptr;
  int offset_first_element_in_bytes;
  int stride_x;
  int stride_y;
} Image;

typedef struct {
  unsigned int krTexIndex;
  unsigned int ktTexIndex;
} MatteTranslucentParam;

typedef struct halide_dimension_t {
  int min, extent, stride;

  uint32_t flags;
} halide_dimension_t;

typedef struct def_ClothVertexData {
  unsigned int vertexID;
  float mass;
  float invmass;
} ClothVertexData;

typedef struct s_texture {
  ulong info_index;
  float2 stretch;
  float2 offset;
} t_texture;

typedef struct whirlpool_ctx {
  unsigned int h[16];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __local unsigned int (*s_Ch)[256];
  __local unsigned int (*s_Cl)[256];

} whirlpool_ctx_t;

typedef struct Layer {

  int numOfNodes;
  Node nodes[1200];

} Layer;

typedef struct whirlpool_hmac_ctx {
  whirlpool_ctx_t ipad;
  whirlpool_ctx_t opad;

} whirlpool_hmac_ctx_t;

typedef struct {
  float Sum;
  float C;
} KahanAccumulator;

typedef struct {
  unsigned long long hash;
  int lock;
  int ply;
  int sd;
  short score;
  short depth;
} ABDADATTE;

typedef struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} clFFT_Dim3;

typedef struct {
  int phiType, region;

  float one_plus_alpha, beta, alpha_beta, one_plus_alpha_beta,
      alpha_by_one_minus_beta, inverse_one_plus_ab_by_diel_ext, kappa, Asq,
      Asq_minus_rnautsq, Asq_minus_rsq, Asq_by_dsq;

} analytical_definitions_struct;

typedef struct Boid {
  float4 separation;
  float4 alignment;
  float4 cohesion;
  float4 goal;
  float4 avoid;
  float4 leaderfollowing;
  float4 float3;
  int num_flockmates;
  int num_nearestFlockmates;
} Boid;

typedef struct {
  Vector sundir;
  Spectrum gain;
  float turbidity;
  float relSize;

  Vector x, y;
  float cosThetaMax;
  Spectrum suncolor;
} SunLight;

typedef struct {
  uint32_t cracked;
} pwsafe_hash;

typedef struct NestedPointer {
  int x;
  struct InnerNestedPointer {
    int *ptrField;
  } inner;
} NestedPointer;

typedef struct {
  unsigned int krTexIndex;
  unsigned int ktTexIndex;
  unsigned int sigmaTexIndex;
} RoughMatteTranslucentParam;

typedef struct {
  int _WclField;
} _WclTypedef;

typedef struct {
  float2 uv;
} TUV;

typedef struct {
  float lo, hi;
} v2float;

typedef struct float5 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
} float5;

typedef struct NestedBool2Inner {
  bool boolField;
} NestedBool2Inner;

typedef struct NestedBool2 {
  int x;
  NestedBool2Inner inner;
} NestedBool2;

typedef struct {
  void **factory;
  void *arg;
} nanos_device_t;

typedef struct {
  float c_Tb;
  float c_TB;
  float c_Tg;
  float c_varInit;
  float c_varMin;
  float c_varMax;
  float c_tau;
  uchar c_shadowVal;
} con_srtuct_t;

typedef struct {
  float x;
} DataStruct;

typedef struct {
  int n, lda, j0;
  short ipiv[32];
} zlaswp_params_t;

typedef struct _PAPhysicsNBodyData {
  float strength, noise;
} PAPhysicsNBodyData;

typedef struct s_lst {
  void *data;
  size_t data_size;
  struct s_lst *next;
} t_lst;

typedef struct _SmokeSimConstants {
  float buoyAlpha;
  float buoyBeta;
  float vorticity;

  float KsDens;
  float KsTemp;

  float KminDens;
  float KmaxDens;
  float KminTemp;
  float KmaxTemp;

  float KdrDens;
  float KdrTemp;
  float KdissipateDens;
  float KdissipateTemp;

  float KpressureJacobiPoissonAlpha;
  float KpressureJacobiPoissonInvBeta;

  float ambiantTemperature;

  float deltaTimeInSeconds;

  float maxDeltaTimeInSeconds;
  float sourceDistributionAlpha;
  float sourceDistributionBeta;
  float2 spoutCenter;
  float2 spoutInvExtent;
  float2 sourceCenter;
  float4 sourceVelocity;
} SmokeSimConstants;

typedef struct PartialKeyFrom3To6 {
  unsigned char partialKeyFrom3To6[4];
} PartialKeyFrom3To6;

typedef struct {

  float4 clippingPlaneCenter;
  Normal clippingPlaneNormal;

  float lensRadius;
  float focalDistance;
} ProjectiveCamera;

typedef struct {
  ProjectiveCamera projCamera;

  float screenOffsetX, screenOffsetY;
  float fieldOfView;

} PerspectiveCamera;

typedef struct cram_md5 {
  unsigned int user[16];

} cram_md5_t;

typedef struct {
  float v[2];
} type_simbolo;

typedef struct {
  float ref_coords_const[3 * 8];
  float rotbonds_moving_vectors_const[3 * 16];
  float rotbonds_unit_vectors_const[3 * 16];
  float ref_orientation_quats_const[4 * 4];
} kernelconstant_conform;

typedef struct {
  float4 corner0;
  float4 cornerx;
  float4 cornery;
} plane_pts;

typedef struct struct_char_x2 {
  char x, y;
} struct_char_x2;

typedef struct __attribute__((packed)) {
  int a;
  int b;
  float c;
} UDD;

typedef struct mwc_s {
  unsigned int w;
} mwc_t;

typedef struct {
  float3 force;
  float3 torque;
} RigidBodyForces;

typedef struct keyboard_layout_mapping {
  unsigned int src_char;
  int src_len;
  unsigned int dst_char;
  int dst_len;

} keyboard_layout_mapping_t;

typedef struct {
  union {
    unsigned char c[8][8][sizeof(int)];
    int v[8][8];
  } xkeys;
} opencl_DES_bs_transfer;

typedef struct {
  float real;
  float imag;
} bh_complex64;

typedef struct mywallet_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} mywallet_tmp_t;

typedef struct b3RigidBodyData b3RigidBodyData_t;

struct b3RigidBodyData {
  float4 m_pos;
  float4 m_quat;
  float4 m_linVel;
  float4 m_angVel;

  int m_collidableIdx;
  float m_invMass;
  float m_restituitionCoeff;
  float m_frictionCoeff;
};

typedef struct vc {
  unsigned int salt_buf[32];
  unsigned int data_buf[112];
  unsigned int keyfile_buf[16];
  unsigned int signature;

  keyboard_layout_mapping_t keyboard_layout_mapping_buf[256];
  int keyboard_layout_mapping_cnt;

  int pim_multi;
  int pim_start;
  int pim_stop;

} vc_t;

typedef struct {
  int size;
  int uniqueElements;
} PermutationSizeContext;

typedef struct {
  float thr_r;
  float thr_g;
  float thr_b;
  float gain;
} CLRgbTnrConfig;

typedef struct {
  float16 lo, hi;
} v2float16;

typedef struct {
  uchar data[32];
} sparse_sign;

typedef struct {
  double x;
  double y;
} af_cdouble;

typedef struct single_array_element_struct_arg {
  int i[4];
} single_array_element_struct_arg_t;

typedef struct {
  int s[1];
} Semaphore;

typedef struct {
  float sinTheta, cosTheta, uScale, vScale, uDelta, vDelta;
} UVMapping2DParam;

typedef struct {
  Semaphore semaphore[2][2];
} Semaphores;

typedef struct CL_AABB {
  float3 point_max;
  float3 point_min;
} CL_AABB;

typedef struct _light_t {
  float4 pos_and_radius;
  float4 col_and_brightness;
  float4 dir_and_spot;
} light_t;

typedef struct KernelFilm {
  float exposure;
  int pass_flag;
  int pass_stride;
  int use_light_pass;

  int pass_combined;
  int pass_depth;
  int pass_normal;
  int pass_motion;

  int pass_motion_weight;
  int pass_uv;
  int pass_object_id;
  int pass_material_id;

  int pass_diffuse_color;
  int pass_glossy_color;
  int pass_transmission_color;
  int pass_subsurface_color;

  int pass_diffuse_indirect;
  int pass_glossy_indirect;
  int pass_transmission_indirect;
  int pass_subsurface_indirect;

  int pass_diffuse_direct;
  int pass_glossy_direct;
  int pass_transmission_direct;
  int pass_subsurface_direct;

  int pass_emission;
  int pass_background;
  int pass_ao;
  int pass_pad1;

  int pass_shadow;
  float pass_shadow_scale;
  int filter_table_offset;
  int pass_pad2;

  int pass_mist;
  float mist_start;
  float mist_inv_depth;
  float mist_falloff;
} KernelFilm;

typedef struct Voxel {

  uchar3 float3;
} Voxel;

typedef struct _NclParamRecList {
  int n_elements;
  struct _NclSymbol *fpsym;
  struct _NclParamRec *the_elements;
} NclParamRecList;

typedef struct struct_char_arr32 {
  char arr[32];
} struct_char_arr32;

typedef struct {
  float4 Sum;
  float4 C;
} KahanAccumulator4;

typedef struct {
  SUM_SSE sum_sse[9];
  char dummy[952];
} subpel_sum_sse;

typedef struct {
  int16 s0, s1, s2, s3;
} v4int16;

typedef struct MPCL_GridKernelData {

  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernels[2];
  cl_mem element;
  cl_mem type;
  cl_mem update;
  cl_mem val;
  cl_mem buf;
  cl_mem inter_x, inter_y, inter_z;
  cl_mem coef_x, coef_y, coef_z;
  cl_mem rhoc;
  cl_mem bound;
  cl_mem cx, cy, cz;
  int local_coef;
} MPCL_GridKernelData;

typedef struct streebog512_ctx_vector {
  unsigned long long h[8];

  unsigned long long s[8];

  unsigned long long n[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __constant unsigned long long (*s_sbob_sl64)[256];

} streebog512_ctx_vector_t;

typedef struct DS2 {
  float2 x, y;
} DS2;

typedef struct _ray_chunk_t {
  unsigned int hash, base, size;
} ray_chunk_t;

typedef struct {
  char data[6];
} ddr_weight;

typedef struct {
  unsigned int texIndex;
  float minVal, maxVal;
} ClampTexParam;

typedef struct {
  uchar v[16];
} gpg_hash;

typedef struct {
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
} mt_struct_stripped;

typedef struct {
  double *buf;
  int dim[3];
} __PSGrid3DDoubleDev;

typedef struct heap {
  char *mem;
  unsigned int size;
} heap;

typedef struct {
  float3 start;
  float3 direction;
  float minScan;
  float maxScan;
  bool binaryEnable;
  bool invertMode;
} sRayMarchingIn;

typedef struct vec {
  float x, y, z;
} vec;

typedef struct s_plane {
  float3 pos;
  float tex_scale;
} t_plane;

typedef struct {
  const char *str;
  int len;
} StringRef;

typedef struct {
  data_16_wng lane[32];
} channel_vec_16_wng;

typedef struct {
  int lower;
  int upper;
  int step;
  bool last;
  bool wait;
  int chunk;
  int stride;
  int thid;
  int threads;
  void *args;
} nanos_loop_info_t;

typedef struct pbkdf2_sha512_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[16];
  unsigned long long out[16];

} pbkdf2_sha512_tmp_t;

typedef struct KernelBVH {

  int root;
  int attributes_map_stride;
  int have_motion;
  int have_curves;
  int have_instancing;

  int pad1, pad2, pad3;
} KernelBVH;

typedef struct myUnpackedStruct2 {
  char2 vec;
} testStruct2;

typedef struct simple_brdf_data {
  float3 emission;
  float3 albedo;
  float reflectivity;
  float transparency;
  float specularity;
} simple_brdf_data;

typedef struct _pass_info_t {
  int index, rand_index;
  int iteration, bounce;
  pass_settings_t settings;
} pass_info_t;

typedef struct pbkdf2_sha1 {
  unsigned int salt_buf[16];

} pbkdf2_sha1_t;

typedef struct {
  int field;
} CRTypedef;

typedef struct {
  float8 vector[4];
} macroblock_vectors_t;

typedef struct _tri_accel_t {
  float nu, nv;
  float np;
  float pu, pv;
  int ci;
  float e0u, e0v;
  float e1u, e1v;
  unsigned int mi, back_mi;
} tri_accel_t;

typedef struct itunes_backup {
  unsigned int wpky[10];
  unsigned int dpsl[5];

} itunes_backup_t;

typedef struct {
  unsigned int v[256 / sizeof(unsigned int)];
} odf_sha_key;

typedef struct s_msg {
  char id;
  size_t size;
  char *data;
  int error;
} t_msg;

typedef struct {
  size_t n;
  unsigned int state;
} StackEntry;

typedef struct __VSSpriteout VSSpriteOut;

struct __GSSpriteOut {
  float4 position;
  float2 textureUV;
};

typedef struct {

  float charge;
  float8 dipole_moment;
  float8 quadrupole_cross_terms;
  float8 quadrupole_trace_terms;

} node_moment_t;

typedef struct {
  StackEntry entries[100];
  int size;
} Stack;

typedef struct {
  Stack indexStack[100];
  int pointer;
} StackVector;

typedef struct {
  unsigned int texIndex;
} RandomTexParam;

typedef struct tm_unz_s {
  unsigned int tm_sec;
  unsigned int tm_min;
  unsigned int tm_hour;
  unsigned int tm_mday;
  unsigned int tm_mon;
  unsigned int tm_year;
} tm_unz;

typedef struct unz_file_info_s {
  ulong version;
  ulong version_needed;
  ulong flag;
  ulong compression_method;
  ulong dosDate;
  ulong crc;
  ulong compressed_size;
  ulong uncompressed_size;
  ulong size_filename;
  ulong size_file_extra;
  ulong size_file_comment;

  ulong disk_num_start;
  ulong internal_fa;
  ulong external_fa;

  tm_unz tmu_date;
} unz_file_info;

typedef struct luks {
  int hash_type;
  int key_size;
  int cipher_type;
  int cipher_mode;

  unsigned int ct_buf[128];

  unsigned int af_src_buf[((512 / 8) * 4000) / 4];

} luks_t;

typedef struct {
  float4 point;
  float4 normal;
  float4 bary;
  float dist;
} Hit;

typedef struct {
  unsigned int iterations;
  unsigned int outlen;
  unsigned int skip_bytes;
  uchar length;
  uchar salt[256];
} pbkdf2_salt;

typedef struct {
  pbkdf2_salt pbkdf2;
  unsigned char iv[8];
  unsigned char ct[256];
} keychain_salt;

typedef struct def_Fluid {
  float kernelRadius;
  unsigned int numSubSteps;
  float restDensity;
  float deltaTime;
  float epsilon;
  float k;
  float delta_q;
  unsigned int n;
  float c;
} Fluid;

typedef struct {
  __global uint8_t *output;
  uint32_t range;
  uint32_t bottom;
  int bit_count;
  uint32_t count;
} vp8_bool_encoder;

typedef struct office2007_tmp {
  unsigned int out[5];

} office2007_tmp_t;

typedef struct differential3 {
  float3 dx;
  float3 dy;
} differential3;

typedef struct {
  uchar layer;
  ushort c_vec;

  bool padding[(3 + 5 - 1) + 3];
  short h;
  short feature_read_w_vec_header;
  short feature_read_w_inc_header;

  bool conv_start;
  bool conv_done[(3 + 5 - 1)];

  bool filter_loading;
  int filter_read_addr;
  char filter_read_fw_vec;
  bool filter_read_page;
  bool filter_loading_conv_idle;

} SequencerOutput;

typedef struct {
  float direction;
  float index_to_physical_point;
  float physical_point_to_index;
  float spacing;
  float origin;
  unsigned int size;
} GPUImageBase1D;

typedef struct TfLiteFloatArray {
  int size;

  float data[];

} TfLiteFloatArray;

typedef struct {

  float3 hit_pt;

  float3 surface_normal;
} hit_info_t;

typedef struct {
  float *buf;
  int dim[3];
} __PSGrid3DFloatDev;

typedef struct {

  uchar activity[2][2];

  uchar fullActivity[2][2];
  bool sequenceSegment;
  bool sequenceSegmentQueued;
  bool hasQueuedChanges;

  float activeDutyCycle;
} Segment;

typedef struct pixel_type {
  float r;
  float g;
  float b;
  float x;
} pixel;

typedef struct arg_aos_struct_type {
  pixel *src;
  pixel *dst;
  int start_index;
  int end_index;
} args_aos;

typedef struct {
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

typedef struct {
  int frame_id;
  int width;
  int height;
  float trim_ratio;
  float proj_mat[9];
} CLWarpConfig;

typedef struct DistanceID_t {
  float distance;
  int id;
} DistanceID;

typedef struct {
  unsigned int dataIndex;
} HitPointAlphaTexParam;

typedef struct {
  unsigned int dataIndex;
} HitPointVertexAOVTexParam;

typedef struct inter_pred_param_for_gpu {
  int src_stride;
  int filter_x_mv;
  int filter_y_mv;
} INTER_PRED_PARAM_GPU;

typedef struct {
  float r_gain;
  float gr_gain;
  float gb_gain;
  float b_gain;
} CLWBConfig;

typedef struct _complex_number {
  float real;
  float img;
} ComplexNumber;

typedef struct {
  union {

    struct {
      float bboxMin[3];
      float bboxMax[3];
    } bvhNode;
    struct {
      unsigned int entryIndex;
    } entryLeaf;
  };

  unsigned int nodeData;
  int pad;
} IndexBVHArrayNode;

typedef struct {
  unsigned int buffer[16];
} T_Lump64;

typedef struct {
  float v[(3 + 5 - 1)];
} OutputWidthVector;

typedef struct {
  OutputWidthVector data[16];
} ReluOutput;

typedef struct {
  unsigned int imageMapIndex;
  unsigned int distributionOffset;
  int useVisibilityMapCache;
} InfiniteLightParam;

typedef struct {
  double cb_kernelParam1;
  double cb_kernelParam2;
  unsigned int cb_instanceLength;
  unsigned int cb_instanceCount;
  unsigned int cb_classIndex;
  unsigned int cb_kernel;
  double cb_param1;
  double cb_param2;
  int cb_ind1;
  int cb_ind2;
} SharedBuffer;

typedef struct single_element_struct_arg {
  int i;
} single_element_struct_arg_t;

typedef struct {
  int m_nConstraints;
  int m_start;
  int m_batchIdx;
  int m_nSplit;

} ConstBuffer;

typedef struct {
  float x;
  float y;
  float vx;
  float vy;
  float r;
} LParticle;

typedef struct fmt {
  const char *key;
  const char *value;
} fmt_t;

typedef struct telegram_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[35];
  unsigned int out[35];

} telegram_tmp_t;

typedef struct float15 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
  float sd;
  float se;
} float15;

typedef struct {
  T_Lump64 lump[2 * 8];
} T_Block;

typedef struct {
  __global float4 *input_min, *input_max;
  __global float4 *output_min, *output_max;
} GlobalLimits;

typedef struct def_ClothSimParams {
  unsigned int numSubSteps;
  float deltaTime;
  float k_stretch;
  float k_bend;
} ClothSimParams;

typedef struct bitlocker_tmp {
  unsigned int last_hash[8];
  unsigned int init_hash[8];

} bitlocker_tmp_t;

typedef struct def_Box {
  float mass;
  float3 halfDimensions;

  float3 position;
  float3 velocity;
  float3 impulses;
  float3 torques;
} Box;

typedef struct kernel_dispatch_packet_s {
  uint16_t header;
  uint16_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved_device_mem_ptr;
  uint64_t completion_signal;
} kernel_dispatch_packet_t;

typedef struct sha512_ctx {
  unsigned long long h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];
  unsigned int w4[4];
  unsigned int w5[4];
  unsigned int w6[4];
  unsigned int w7[4];

  int len;

} sha512_ctx_t;

typedef struct {
  unsigned int length;
  unsigned int iterations;
  uchar salt[16];
} sevenzip_salt;

typedef struct {
  ddr_weight vec[16];
} ddr_weight_wng;

typedef struct pkzip pkzip_t;

__constant unsigned int crc32tab[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};

typedef struct sha1_ctx_vector {
  unsigned int h[5];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha1_ctx_vector_t;

typedef struct {
  float4 colorInfo;
  float4 sceneInfo;
} PostProcessingBuffer;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_R __attribute__((aligned(4)));
  unsigned int idx_Fresnel;
} SpecularRElem;

typedef struct {
  unsigned int bucketIndex, pixelOffset, passOffset, pass;

  Seed rngGeneratorSeed;
  unsigned int rngPass;
  float rng0, rng1;
} SobolSample;

typedef struct __attribute__((aligned(1))) {
  uchar texType;
  uchar procedureType;
} ProceduralTextureHead;

typedef struct __attribute__((aligned(16))) {
  ProceduralTextureHead head;
  float3 c[2] __attribute__((aligned(16)));
} Float3CheckerBoardTexture;

typedef struct DS4 {
  float2 x, y, z, w;
} DS4;

typedef struct PVPatchStrides_ {
  int sx, sy, sf;
} PVPatchStrides;

typedef struct {
  float _0;
  float _1;
} Tuple_float_float;

typedef struct Vec4_t {
  float x;
  float y;
  float z;
  float w;
} Vec4;

typedef struct Filter {

  float weights[49];
  float bias;

} Filter;

typedef struct ConvLayer {

  int numOfFilters;
  Filter filters[10];

} ConvLayer;

typedef struct {
  unsigned int A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, AA, AB;
  unsigned int B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, BA, BB, BC, BD, BE, BF;
  unsigned int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, CA, CB, CC, CD, CE, CF;
  unsigned int Wlow, Whigh;
} shabal_context_t;

typedef struct Individual {

  unsigned char a[64];

  float fitness[4];

  int nSelFeatures;

  int rank;

  float crowding;

} Individual;

typedef struct {
  unsigned int tex1Index, tex2Index;
} AddTexParam;

typedef struct whirlpool_ctx_vector {
  unsigned int h[16];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __local unsigned int (*s_Ch)[256];
  __local unsigned int (*s_Cl)[256];

} whirlpool_ctx_vector_t;

typedef struct {
  float16 direction;
  float16 index_to_physical_point;
  float16 physical_point_to_index;
  float3 spacing;
  float3 origin;
  uint3 size;
} GPUImageBase3D;

typedef struct diskcryptor_esalt {
  unsigned int salt_buf[512];

} diskcryptor_esalt_t;

typedef struct {
  Direction direction[64][2];
} GlobalDirection;

typedef struct {
  float deltaTime;
  float particleRadius;
  float cellSize;
  float mass;
  float viscosity;
  float pressure;
  float density;
  float spacing;
  float stiffness;
  float viscosityConstant;
  float pressureConstant;
  float kernelConstant;
  float velocitylimit;
} System;

typedef struct Sphere {
  float4 pos_and_r;
  float4 float3;
  unsigned int texture_index;
  float reflection;
  float refraction;
  float eta;
} Sphere;

typedef struct {
  unsigned int key[32 / 4];
} sevenzip_hash;

typedef struct {
  int x;
  int y __attribute((aligned(16)));
} s;

typedef struct {
  int float3;
  float ls;
} RT_Emissive;


typedef struct {
  float x;
  float y;
} decimal2;

typedef struct {
  float x;
  float y;
  float z;
  float w;
} decimal4;

typedef struct {
  int x;
  int y;
  int w;
  int h;
} lbp_rect;

typedef struct _PAPhysicsGravityData {
  float strength, noise;
  float mass;
} PAPhysicsGravityData;

typedef struct {
  int P;
  unsigned int Length;
  char FileID[16];
  char U[32];
  char O[32];
} PDFParams;

typedef union {
  int i;
  float f;
} union_int_float_t;

typedef union {
  float vecAccess[3];
  Vector vec;
} VectorAccess;

typedef struct intintIdentitySentinelPerfectCLHash_TableData {
  int hashID;
  unsigned int numBuckets;
  char compressFuncData;
  int emptyValue;
} intintIdentitySentinelPerfectCLHash_TableData;

typedef struct clb2Rotation {
  float s;
  float c;
} clb2Rotation;

typedef struct b2clContactFeature {
  uchar indexA;
  uchar indexB;
  uchar typeA;
  uchar typeB;
} b2clContactFeature;

typedef struct CSGOperation_s {
  int type;
  int brushFunction;
  int brushMaterial;
  float rotateY;
  float4 origin;
  float4 dimensions;
} CSGOperation;

typedef union b2clContactID {
  b2clContactFeature cf;
  unsigned int key;
} b2clContactID;

typedef struct b2clManifoldPoint {
  float2 localPoint;
  float normalImpulse;
  float tangentImpulse;
  b2clContactID id;
  float dummy;
} b2clManifoldPoint;

typedef struct {
  unsigned int krIndex;
} FresnelColorParam;

typedef struct _vx_tensor_t *vx_tensor;
typedef enum _vx_bool_e {

  vx_false_e = 0,

  vx_true_e,
} vx_bool_e;

typedef struct {
  uint3 ActualLocalSize;
  uint3 WalkerDimSize;
  uint3 WalkerStartPoint;
} IGIL_WalkerData;

typedef struct {
  float real;
  float imag;
} clFFT_Complex;

typedef struct {
  uint32_t buffer[32];
  uint32_t buflen;
} sha512_ctx;

typedef struct {
  float initpos;
  float kd;
} GPUSpring2;

typedef struct radix_config {
  unsigned int radices;
  unsigned int blocks;
  unsigned int gpb;
  unsigned int tpg;
  unsigned int epg;
  unsigned int rpb;
  unsigned int mask;
  unsigned int l_val;
  unsigned int tpb;
  unsigned int size;
} configuration;

typedef struct atmi_implicit_args_s {
  unsigned long offset_x;
  unsigned long offset_y;
  unsigned long offset_z;
  unsigned long hostcall_ptr;
  char num_gpu_queues;
  unsigned long gpu_queue_ptr;
  char num_cpu_queues;
  unsigned long cpu_worker_signals;
  unsigned long cpu_queue_ptr;
  unsigned long kernarg_template_ptr;

} atmi_implicit_args_t;

typedef struct ethereum_pbkdf2 {
  unsigned int salt_buf[16];
  unsigned int ciphertext[8];

} ethereum_pbkdf2_t;

typedef struct {
  unsigned int total[2];
  unsigned int state[8];
  uchar buffer[64];
} SHA256_CTX;

typedef struct {
  ulong t;
  SHA256_CTX ctx;
  unsigned int len;
  ushort buffer[255];
} sevenzip_state;

typedef struct _data_type {
  int value[3];
} data_type;

typedef struct pw_idx {
  unsigned int off;
  unsigned int cnt;
  unsigned int len;

} pw_idx_t;

typedef struct flexible_array {
  int i;
  int flexible[];
} flexible_array;

typedef struct {
  bool mandatory_creation;
  bool tied;
  bool clear_chunk;
  bool reserved0;
  bool reserved1;
  bool reserved2;
  bool reserved3;
  bool reserved4;
} nanos_wd_props_t;

typedef struct TINYMT32WP_T {
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
  unsigned int s3;
  unsigned int mat1;
  unsigned int mat2;
  unsigned int tmat;
} tinymt32wp_t;

typedef struct {
  float4 m_worldPos[4];
  float4 m_worldNormal;
  unsigned int m_coeffs;
  int m_batchIdx;

  int m_bodyA;
  int m_bodyB;
} Contact4;

typedef struct __attribute__((packed)) _GmmParams {

  int2 size;

  int k;

  float maha_thresh;

  float alpha;

  float w_init;

  float var_init;

  float var_min;

  float density_thresh;
} GmmParams;

typedef struct tm_zip_s {
  unsigned int tm_sec;
  unsigned int tm_min;
  unsigned int tm_hour;
  unsigned int tm_mday;
  unsigned int tm_mon;
  unsigned int tm_year;
} tm_zip;

typedef struct {
  float fPowKLow;
  float fPowKHigh;
  float fPow35;
  float fFStops;
  float fFStopsInv;
  float fPowExposure;
  float fGamma;
  float fPowGamma;
  float fDefog;
} CHDRData;

typedef struct __attribute__((packed)) {
  uint4 dimensions_;
  uchar numChannels_;
  uchar numBitsPerChannel_;
  uchar isSigned_;
  uchar isFloat_;
  signed int data_;
} volume_t;

typedef struct {
  double4 lo, hi;
} v2double4;

typedef struct General {
  float4 backgroundColor;
  float4 offLensColor;
  float rayStartOffset;
  int maxRayIterations;
  int maxPathLength;
  int pathSampleCount;
  int multiSampleCountSqrt;
  int timeSampleCount;
  int cameraSampleCountSqrt;
  int visualizeDepth;
  float visualizeDepthMaxDistance;
  int enableNormalMapping;
} General;

typedef struct t_keypoint {
  float4 kp;
  unsigned char desc[128];
} t_keypoint;

typedef struct _svm_parameter {
  int svm_type;
  int kernel_type;
  int degree;
  float gamma;
  float coef0;
} svm_parameter;

typedef struct {
  Spectrum radiance;

  unsigned int pixelIndex;

  float u[2];
} Sample;

typedef struct {
  tm_zip tmz_date;
  ulong dosDate;

  ulong internal_fa;
  ulong external_fa;
} zip_fileinfo;

typedef struct clb2Position {
  float cx;
  float cy;
  float a;
} clb2Position;

typedef struct sm_s {
  unsigned int w;
  unsigned int z;
} sm_t __attribute__((aligned(8)));

typedef struct {
  uint32_t size[1024];
  uint64_t d[10 * 1024];
} mpzcl_t;

typedef struct {
  unsigned int state;
  unsigned int depth;
  Spectrum throughput;
} PathState;

typedef struct {

  Seed seed;

  Sample sample;

  PathState pathState;
} GPUTask;

typedef struct {
  unsigned int textureIndex;
  unsigned int moduloIndex;
} ModuloTexParam;

typedef struct {
  unsigned int uiTransferId;
  unsigned int uiBufferId;
  unsigned long long ullBufferBusAddress;
  unsigned long long ullMarkerBusAddress;
} FrameData;

typedef struct _mesh_t {
  unsigned int node_index, node_count;
  unsigned int tris_index, tris_count;
} mesh_t;

typedef struct {
  uint32_t buflen;
  buffer_32 buffer[16];
} sha256_ctx;

typedef struct clrngLfsr113Stream_ clrngLfsr113Stream;

typedef struct {

  cl_uint g[4];

} clrngLfsr113StreamState;

typedef struct sphere_object_data {
  float3 position;
  float radius;
} sphere_object_data;

typedef struct b3GpuFace b3GpuFace_t;
struct b3GpuFace {
  float4 m_plane;
  int m_indexOffset;
  int m_numIndices;
  int m_unusedPadding1;
  int m_unusedPadding2;
};

struct clrngLfsr113HostStream_ {
  clrngLfsr113StreamState current;
  clrngLfsr113StreamState initial;
  clrngLfsr113StreamState substream;
};

typedef struct {
  float16 viewProjMatrix;
  float3 position;
  float3 lookVector;

  float3 ambient;
  float3 diffuse;
  float3 specular;

  bool hasShadows;

} cl_dirlight;

typedef struct {
  double T;
  double A;
  double F;
  int start;
  int end;
} Window;

typedef struct _PAPhysicsBoidsData {
  float strength, noise;
} PAPhysicsBoidsData;

typedef struct MatrixFloatPrivate {

  __private float *data;
  size_t rowCount;
  size_t columnCount;

} MatrixFloatPrivate;

typedef struct {

  float4 m_deltaLinearVelocity;
  float4 m_deltaAngularVelocity;
  float4 m_angularFactor;
  float4 m_linearFactor;
  float4 m_invMass;
  float4 m_pushVelocity;
  float4 m_turnVelocity;
  float4 m_linearVelocity;
  float4 m_angularVelocity;

  union {
    void *m_originalBody;
    int m_originalBodyIndex;
  };
  int padding[3];

} b3GpuSolverBody;

typedef struct {
  double *data;
  unsigned int *shape;
  unsigned int *strides;
  unsigned int ndims;
} ndarray;

typedef struct s_fir_params {
  int intracksize;
  int decimation;
} t_fir_params;

typedef struct {
  float x;
  float y;
} af_cfloat;

typedef struct float6 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
} float6;

typedef struct _NclMultiDValnclfileDataPart {
  char *space_holder;
} NclMultiDValnclfileDataPart;

typedef struct __attribute__((aligned(64))) GpuHidHaarStageClassifier {
  int count __attribute__((aligned(4)));
  float threshold __attribute__((aligned(4)));
  int two_rects __attribute__((aligned(4)));
  int reserved0 __attribute__((aligned(8)));
  int reserved1 __attribute__((aligned(8)));
  int reserved2 __attribute__((aligned(8)));
  int reserved3 __attribute__((aligned(8)));
} GpuHidHaarStageClassifier;

typedef struct {
  int ind1;
  int ind2;
  int ind3;
} evalStruct;

typedef struct {
  uint8_t v[87];
  uint32_t length;
} pwsafe_pass;

typedef struct _NclFileAttInfoList {
  struct _NclFAttRec *the_att;
  struct _NclFileAttInfoList *next;
} NclFileAttInfoList;

typedef struct {
  unsigned int workDim;
  unsigned int globalSize[3];
  unsigned int globalID[3];
  unsigned int localSize[3];
  unsigned int localID[3];
  unsigned int numGroups[3];
  unsigned int groupID[3];
} work_item_data;

typedef struct b2clMat33 {
  float ex[3];
  float ey[3];
  float ez[3];
} b2clMat33;

typedef struct TreeNode {
  int leftChild;
  short4 feature;
  short threshold;
  float histogram[(22)];
} TreeNode;

typedef struct b2clPrismaticJointData {

  float localAnchorA[2];
  float localAnchorB[2];
  float localXAxisA[2];
  float localYAxisA[2];
  float referenceAngle;
  float lowerTranslation;
  float upperTranslation;
  float maxMotorForce;
  float motorSpeed;
  int enableLimit;
  int enableMotor;
  int limitState;

  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  float axis[2], perp[2];
  float s1, s2;
  float a1, a2;
  b2clMat33 K;
  float motorMass;
} b2clPrismaticJointData;

typedef struct {
  float2 pos;
  float2 prev;
  float2 force;
  float mass;
  char isLocked;
} Particle2;

typedef struct MatrixFloatLocal {

  __local float *data;
  size_t rowCount;
  size_t columnCount;

} MatrixFloatLocal;

typedef struct {
  uint64_t base;
  uint64_t stride;
  uint32_t head;
  uint32_t tail;
} clIndexedQueue_64;

typedef struct {
  int residualOrder;
  int samplesOffs;
  int shift;
  int cbits;
  int size;
  int type;
  int obits;
  int blocksize;
  int coding_method;
  int residualOffs;
  int wbits;
  int abits;
  int porder;
  int headerLen;
  int encodingOffset;
} FLACCLSubframeData;

typedef struct {
  float x;
  float y;
} polygonPoint;

typedef struct {
  int firstObject;
  int endObject;
} CollisionObjectIndices;

typedef struct {
  uint8_t length;
  uint8_t salt[115];
  uint32_t rounds;
  union blob {
    uint64_t qword[24 / 8];
    uint8_t chr[24];
  } blob;
} fvde_salt_t;

typedef struct {
  int chrom[11 + 1];
  double x[11 + 1], objective, fitness;
  int parent1, parent2;
  int xsite;
} ind_t;

typedef struct s_rectangle {
  float3 pos;
  float h;
  float w;
  float tex_scale;
} t_rectangle;

typedef struct def_VoxelGridInfo {

  uint3 grid_dimensions;

  unsigned int total_grid_cells;

  float grid_cell_size;

  float3 grid_origin;

  unsigned int max_cell_particle_count;
} VoxelGridInfo;

typedef struct {
  FLACCLSubframeData data;
  int coefs[32];
} FLACCLSubframeTask;

typedef struct {
  float4 lo, hi;
} v2float4;

typedef struct sha224_ctx_vector {
  unsigned int h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha224_ctx_vector_t;

typedef struct struct_char_x8 {
  char x, y, z, w;
  char a, b, c, d;
} struct_char_x8;

typedef struct {
  float *real;
  float *imag;
} clFFT_SplitComplex;

typedef struct {
  unsigned int lightIndex;
  float pickPdf;

  float directPdfW;

  Spectrum lightRadiance;

  Spectrum lightIrradiance;

  unsigned int lightID;
} DirectLightIlluminateInfo;

typedef struct {
  DPTYPE_VEC lane[8];
} DPTYPE_SCAL_VEC;

typedef struct OCL_object {
  cl_int status;
  cl_event *events;
  cl_kernel _kernel;
  cl_context context;
  cl_program *program;
  cl_device_id device_id;
  cl_platform_id platform_id;
  cl_command_queue *queues;
  int queue_count;
  char is_initialized;
} OCL_object;

typedef struct sha256aix_tmp {
  unsigned int ipad[8];
  unsigned int opad[8];

  unsigned int dgst[8];
  unsigned int out[8];

} sha256aix_tmp_t;

typedef struct b2clBodyStatic {
  float2 m_localCenter;
  float m_invMass;

  float m_invI;
  float m_linearDamping;
  float m_angularDamping;
  float m_gravityScale;
  float m_type;
  int m_connectedBodyIndices[8];
  int m_bIsBullet;
  int dummy;
} b2clBodyStatic;

typedef struct {
  union {
    int4 vecInt;
    int ints[4];
  };
} Int4CastType;

typedef struct {
  float lambda, epsilon_data, epsilon_reg, tau, sigma;
  int2 sdim, ldim;
  float scale_factor;
  float sensor_sigma;

  float dual_p_denom;
  float dual_q_denom;
  float sf_2;
} super_res_params;

typedef struct {
  float4 m_row[3];
} b3Mat3x3;

typedef struct _cavity {
  int occupancy;
  double pos[3];
} cavity_t;

typedef struct {
  int field;
} PTypedef;

typedef struct {
  PTypedef pstruct;
} PPTypedef;

typedef struct {
  float2 a;
  float2 b;
} line2;

typedef struct sha512crypt_tmp {
  unsigned long long l_alt_result[8];
  unsigned long long l_p_bytes[2];
  unsigned long long l_s_bytes[2];

  unsigned int alt_result[16];
  unsigned int p_bytes[64];
  unsigned int s_bytes[64];

} sha512crypt_tmp_t;

typedef struct {
  uint64_t v[8];
} sha512_hash;

typedef struct {
  unsigned int total;
  unsigned int state[5];
  uchar buffer[64];
} sha1_context;

typedef struct {
  double *buf;
  int dim[1];
} __PSGrid1DDoubleDev;

typedef struct WALL {
  float3 point;
  float3 normal;
} WALL;

typedef struct {

  uint4 P[1];

} scrypt_tmp_t;

typedef struct plain {
  unsigned long long gidvid;
  unsigned int il_pos;
  unsigned int salt_pos;
  unsigned int digest_pos;
  unsigned int hash_pos;
  unsigned int extra1;
  unsigned int extra2;

} plain_t;

typedef struct {
  float3 row0;
  float3 row1;
  float3 row2;
} homography;
typedef union if32_ {
  uint32_t u;
  int s;
  float f;
} if32;

typedef union {
  unsigned char h1[16];
  unsigned int h4[4];
  ulong h8[2];
} hash16_t;

typedef union {
  unsigned int h4[32];
  ulong h8[16];
  uint4 h16[8];
  ulong2 hl16[8];
  ulong4 h32[4];
} lyraState_t;

typedef union {
  struct {
    unsigned int xi[7];
    unsigned int padding;
  } slot;
  uint8 ui8;
  uint4 ui4[2];
  uint2 ui2[4];
  unsigned int ui[8];

} slot_t;

typedef union vconv64 {
  unsigned long long v64;

  struct {
    unsigned int a;
    unsigned int b;

  } v32;

  struct {
    unsigned short a;
    unsigned short b;
    unsigned short c;
    unsigned short d;

  } v16;

  struct {
    unsigned char a;
    unsigned char b;
    unsigned char c;
    unsigned char d;
    unsigned char e;
    unsigned char f;
    unsigned char g;
    unsigned char h;

  } v8;

} vconv64_t;

typedef union KernelArgValue {
  cl_mem mem;
  int ival;
  unsigned char data[4];
} KernelArgValue;

typedef union {
  unsigned int intVal;
  float floatVal;
} int_and_float;

typedef union {
  unsigned char h1[64];
  unsigned int h4[16];
  ulong h8[8];
} hash_t;

typedef union LeadingDimention {
  size_t matrix;
  int vector;
} LeadingDimention;

typedef union {
  float3 solid_color;
} TextureData;

typedef union GPtr {
  __global float *f;
  __global double *d;
  __global float2 *f2v;
  __global double2 *d2v;
  __global float4 *f4v;
  __global double4 *d4v;
  __global float8 *f8v;
  __global double8 *d8v;
  __global float16 *f16v;
  __global double16 *d16v;
} GPtr;

typedef union _nonce_t {
  uint2 uint2_s;
  ulong ulong_s;
} nonce_t;

typedef union {
  uint64_t as_uint64[16];
  uint32_t as_uint32[32];
  hash_t hash;
  struct {
    uint64_t truncated_hash[4];
    salt_t salt;
  };
} hash_block_t;

typedef union {
  ulong xc;
  struct {
    unsigned int x;
    unsigned int c;
  };
} mwc64x_state;

typedef union {
  uint2 b;
  short s[(4)];
} bus_to_short_t;

typedef union {
  long all;
  struct {
    int high;
    unsigned low;
  } s;
} dwords;

typedef union {
  uint4 uint4s[4];
  ulong ulongs[8];
  unsigned int uints[16];
} compute_hash_share;

typedef union {

} gpu_mem;

typedef union {
  uint4 m_int4;
  unsigned int m_ints[4];
} naive_conv;

typedef union {
  unsigned int h[8];
  uint4 h4[2];
  ulong2 hl4[2];
  ulong4 h8;
} hashly_t;

typedef union buffer_64_u {
  uint8_t mem_08[8];
  uint16_t mem_16[4];
  uint32_t mem_32[2];
  uint64_t mem_64[1];
} buffer_64;

typedef union {
  ulong LR;
  struct {
    unsigned int L, R;
  };
} philox2x32_10_state;

typedef union _Color {
  struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
  };
  unsigned int c;
} Color;

typedef union {
  uint2 b;
  float f[(1 << ((4)))];
} bus_to_float_t;

typedef union cvt_bf16_fp32 {
  unsigned int z;
  ushort2 ushortx2;

  float f32;
} cvt_bf16_fp32_t;

typedef union {
  uint2 b;
  unsigned char a[(64)];
} bus_to_u8_t;

typedef union FType {
  unsigned char u[sizeof(cl_double)];
  float f;
  cl_float2 f2;
  cl_double d;
  cl_double2 d2;
} FType;

typedef union {
  struct {
    unsigned int a, b, c, d;
  };
  ulong res;
} tyche_i_state;

typedef struct {
  uint32_t length;
  buffer_64 pass[((23 + 7) / 8)];
} sha512_password;

typedef struct s_Ndividers {

  int Octave;
  int N;
  int OctaveN;
  int Nsegmentsize;
  int Nsegment;
  int Nbasestep;
  int Nstep;
  int localn;
} t_Ndividers;

typedef struct half7 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
} half7;

typedef struct {
  int m_nContacts;
  float m_dt;
  float m_positionDrift;
  float m_positionConstraintCoeff;
} ConstBufferCTC;

typedef struct {
  DPTYPE_SCAL_VEC data[(416 / 32)];
} PE_SCAL_VEC;

typedef struct atmi_tprofile_s {
  unsigned long int dispatch_time;
  unsigned long int ready_time;
  unsigned long int start_time;
  unsigned long int end_time;
} atmi_tprofile_t;

typedef struct Vec5 {
  float data[5];
} Vec5;

typedef struct md5crypt_tmp {
  unsigned int digest_buf[4];

} md5crypt_tmp_t;

typedef struct _PAPhysicsNewtonian {
  float fixed, mass;
  float ax, ay, az;
  float ox, oy, oz;
} PAPhysicsNewtonian;

typedef struct {
  char lane[32];
} channel_scal;

typedef struct {
  uint32_t m_bits[4];
} encoded_block;

typedef struct _observables {
  double energy;
  double coulombic_energy, rd_energy, polarization_energy, vdw_energy,
      three_body_energy;
  double dipole_rrms;
  double kinetic_energy;
  double temperature;
  double volume;
  double N, NU;
  double spin_ratio;
  double frozen_mass, total_mass;
} observables_t;

typedef struct {
  float m[4][4];
} Matrix4x4;

typedef struct vc64_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[32];
  unsigned long long out[32];

  unsigned long long pim_key[32];
  int pim;
  int pim_check;

} vc64_tmp_t;

typedef struct krb5pa_17 {
  unsigned int user[128];
  unsigned int domain[128];
  unsigned int account_info[512];
  unsigned int account_info_len;

  unsigned int checksum[3];
  unsigned int enc_timestamp[32];
  unsigned int enc_timestamp_len;

} krb5pa_17_t;

typedef struct {
  uint32_t hash[((32 + 31) / 32) * 32 / sizeof(uint32_t)];
} crack_t;

typedef struct _FoundStruct {
  unsigned int workIndex;
  unsigned int foundBits[8];
} FoundStruct;

typedef struct b2clClipVertex {
  float2 v;
  b2clContactID id;
} b2clClipVertex;

typedef struct __attribute__((aligned(4))) {

  float uLens[2];
} CameraSample;

typedef struct sols_s {
  unsigned int nr;
  unsigned int likely_invalids;
  uchar valid[11];
  unsigned int values[11][(1 << 9)];
} sols_t;

typedef struct {

  node_moment_t moment;

} node_value_t;

typedef struct {
  int arr[2];
} S1;

typedef struct {
  float reqm_const[2 * 2];
  unsigned int atom1_types_reqm_const[2];
  unsigned int atom2_types_reqm_const[2];
  float VWpars_AC_const[4 * 4];
  float VWpars_BD_const[4 * 4];
  float dspars_S_const[4];
  float dspars_V_const[4];
} kernelconstant_intra;

typedef struct {
  ulong ulongs[32 / sizeof(ulong)];
} hash32_t;

typedef struct intintHash_CompressLCGData {
  long unsigned int a;
  long unsigned int c;
  unsigned int m;
  unsigned int n;
} intintHash_CompressLCGData;

typedef struct {
  cl_mem buffer;
  unsigned int length;
  unsigned int stride;
  size_t datasize;
} vector_buffer;

typedef struct loopfilter_param_ocl {
  int dst_offset;
  int dst_stride;

  int mi_row;
  int mi_rows;
} LOOPFILTER_PARAM_OCL;

typedef struct b2clFixtureStatic {
  float m_friction;
  float m_restitution;
  int m_last_uid;
  int dummy;
} b2clFixtureStatic;

typedef struct OctreeNode {

  float3 float3;

  unsigned int firstChild;
} OctreeNode;

typedef struct {
  float pos_x;
  float pos_y;
} cv_pos;

typedef struct Quad {

  float4 pos;
  float4 norm;
  float4 ke;
  float4 kd;
  float4 ks;
  float4 edge_l;
  float4 edge_w;
  float phong_exponent;

} Quad;

typedef struct {
  float4 m_gravity;
  float4 m_worldMin;
  float4 m_worldMax;

  float m_particleRad;
  float m_globalDamping;
  float m_boundaryDamping;
  float m_collisionDamping;

  float m_spring;
  float m_shear;
  float m_attraction;
  float m_dummy;
} btSimParams;

typedef struct acceleration_ {
  float4 a;
} acceleration_t;

typedef struct ikepsk {
  unsigned int nr_buf[16];
  unsigned int nr_len;

  unsigned int msg_buf[128];
  unsigned int msg_len[128];

} ikepsk_t;

typedef struct intintLCGLinearOpenCompactCLHash_TableData {
  int hashID;
  unsigned int numBuckets;
  intintHash_CompressLCGData compressFuncData;
} intintLCGLinearOpenCompactCLHash_TableData;

typedef struct {
  int cells[9];
} Mat3X3;

typedef struct {
  unsigned int texIndex;
} FresnelApproxKTexParam;

typedef struct {
  unsigned int m_height4[4 * 4 * 6 / 4];

  float m_scale;
} ShapeDeviceData;

typedef struct test_struct {
  int elementA;
  int elementB;
  long elementC;
  char elementD;
  long elementE;
  float elementF;
  short elementG;
  double elementH;
} test_struct;

typedef struct pbkdf2_md5_tmp {
  unsigned int ipad[4];
  unsigned int opad[4];

  unsigned int dgst[32];
  unsigned int out[32];

} pbkdf2_md5_tmp_t;

typedef struct float9 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
} float9;

typedef struct {

  unsigned int node_a_index;

  unsigned int node_b_index;

  unsigned int node_a_interaction_index;

  unsigned int node_b_interaction_index;

  unsigned char can_approx;

  unsigned char can_reduce;

} interaction_t;

typedef struct pstoken {
  unsigned int salt_buf[128];
  unsigned int salt_len;

  unsigned int pc_digest[5];
  unsigned int pc_offset;

} pstoken_t;

typedef struct _stack_t {
  unsigned int u;
  unsigned int c;
  bool d;
  unsigned char k;
} stack_t;

typedef struct {
  int partition_count;
  uint8_t texels_per_partition[4];
  uint8_t partition_of_texel[216];
  uint8_t texels_of_partition[4][216];
  uint64_t coverage_bitmaps[4];
} partition_info;

typedef struct keychain {
  unsigned int data[12];
  unsigned int iv[2];

} keychain_t;

typedef struct __blake2b_state {
  uint64_t h[8];
  uint64_t t[2];
  uint64_t f[2];
  uint8_t buf[2 * 16];
  size_t buflen;
  uint8_t last_node;
} blake2b_state;

typedef struct _Vector3 {
  union {
    struct {
      float x;
      float y;
      float z;
    };
    float data[3];
  };
} Vector3;

typedef struct {
  float2 lo, hi;
} v2float2;

typedef struct {
  float x[12];
} Thing;

typedef struct _Plane {
  Vector3 center;
  Vector3 normal;

  Material material;
} Plane;

typedef struct {

  float Sx, Sy, Sz;

  float Sxy, Sxz, Syz;

  Matrix4x4 R;

  float Tx, Tz;

  float Px, Py, Pz, Pw;

  bool Valid;
} DecomposedTransform;

typedef struct {
  ulong x, c, y, z;
} kiss09_state;

typedef struct {
  unsigned int currentVolumeIndex;
  unsigned int volumeIndexList[8];
  unsigned int volumeIndexListSize;

  int scatteredStart;
} PathVolumeInfo;

typedef struct {
  int i;
  int n;
} program_state_struct;

typedef struct {
  double energy;
  double total_xs;
  double elastic_xs;
  double absorbtion_xs;
  double fission_xs;
  double nu_fission_xs;
} NuclideGridPoint;

typedef struct {
  Semaphores semaphores[64][2];
} GlobalSemaphores;

typedef struct {
  union {
    float4 data;
    float3 ray;
    struct {
      float direction[3];
      unsigned int intersects;
    };
  };
} PRay;

typedef struct {
  unsigned int radianceGroupCount;
  int bcdDenoiserEnable, usePixelAtomics;

  int hasChannelAlpha;
  int hasChannelDepth;
  int hasChannelPosition;
  int hasChannelGeometryNormal;
  int hasChannelShadingNormal;
  int hasChannelMaterialID;
  int hasChannelDirectDiffuse;
  int hasChannelDirectDiffuseReflect;
  int hasChannelDirectDiffuseTransmit;
  int hasChannelDirectGlossy;
  int hasChannelDirectGlossyReflect;
  int hasChannelDirectGlossyTransmit;
  int hasChannelEmission;
  int hasChannelIndirectDiffuse;
  int hasChannelIndirectDiffuseReflect;
  int hasChannelIndirectDiffuseTransmit;
  int hasChannelIndirectGlossy;
  int hasChannelIndirectGlossyReflect;
  int hasChannelIndirectGlossyTransmit;
  int hasChannelIndirectSpecular;
  int hasChannelIndirectSpecularReflect;
  int hasChannelIndirectSpecularTransmit;
  int hasChannelMaterialIDMask;
  unsigned int channelMaterialIDMask;
  int hasChannelDirectShadowMask;
  int hasChannelIndirectShadowMask;
  int hasChannelUV;
  int hasChannelRayCount;
  int hasChannelByMaterialID;
  unsigned int channelByMaterialID;
  int hasChannelIrradiance;
  int hasChannelObjectID;
  int hasChannelObjectIDMask;
  int channelObjectIDMask;
  int hasChannelByObjectID;
  unsigned int channelByObjectID;
  int hasChannelSampleCount;
  int hasChannelConvergence;
  int hasChannelMaterialIDColor;
  int hasChannelAlbedo;
  int hasChannelAvgShadingNormal;
  int hasChannelNoise;
  int hasChannelUserImportance;
} Film;

typedef struct {
  uchar red;
  uchar green;
  uchar blue;
  uchar alpha;
} RGBA;

typedef struct {
  int width;
  int height;
  __global float *elements;
} Matrix;

typedef struct __attribute__((aligned(128))) GpuHidHaarTreeNode {
  int p[3][4] __attribute__((aligned(64)));
  float weight[3];
  float threshold;
  float alpha[3] __attribute__((aligned(16)));
  int left __attribute__((aligned(4)));
  int right __attribute__((aligned(4)));
} GpuHidHaarTreeNode;

typedef struct {
  double x, y;
} Sleef_double2;

typedef struct _measurement {
  float angles[5];
  float x;
  float y;
} measurement;

typedef struct {
  unsigned int texIndex;
} NormalMapTexParam;

typedef struct {

  unsigned int data[128];
  unsigned int front;
  unsigned int tail;
  unsigned int size;
} Queue;

typedef struct __attribute__((aligned(32))) GpuHidHaarClassifier {
  int count __attribute__((aligned(4)));
  GpuHidHaarTreeNode *node __attribute__((aligned(8)));
  float *alpha __attribute__((aligned(8)));
} GpuHidHaarClassifier;

typedef struct {
  unsigned int texIndex;
  unsigned int hueTexIndex, satTexIndex, valTexIndex;
} HsvTexParam;

typedef struct {

  float8 field;

} leaf_field_t;

typedef struct {
  int bla;
  int numElem;
} MyAabbConstDataCL;

typedef struct {
  leaf_field_t field_a;
  leaf_field_t field_b;
} leaf_field_pair_t;

typedef struct clb2Manifold {
  float2 localNormal;
  float2 localPoint;
  float2 localPoints1;
  float2 localPoints2;
  int pointCount;
  int type;
  float radiusA;
  float radiusB;
  float2 localCenterA;
  float2 localCenterB;
} clb2Manifold;

typedef struct {
  int length;
  unsigned char significantBits;
  unsigned char codingPasses;
  unsigned char width;
  unsigned char nominalWidth;
  unsigned char height;
  unsigned char nominalHeight;
  unsigned char stripeNo;
  unsigned char magbits;
  unsigned char subband;
  unsigned char compType;
  unsigned char dwtLevel;
  float stepSize;

  int magconOffset;

  int gpuCoefficientsOffset;
  int gpuCodestreamOffset;
  int gpuSTBufferOffset;
} CodeBlockAdditionalInfo;

typedef struct {
  unsigned int dataIndex;
} HitPointColorTexParam;

typedef struct {
  int _WCL_FIELD;
} _WCL_TYPEDEF;

typedef struct {
  int setup_numOfLayers;
  int setup_b_offset_layers;
  bool setup_classification;
  float setup_minOutputValue;
  float setup_maxOutputValue;
  float setup_learningFactor;
  float setup_lambda;

  int criteria_maxEpochs;
  float criteria_minProgress;
  float criteria_maxGeneralizationLoss;

  int state_epoch;
  int state_b_offset_weights;
  int state_b_offset_values;
  int state_b_size_values;
  int state_b_offset_errors;
  int state_b_size_errors;
  int state_testLine;
  int state_learningLine;

  float neuralNetwork_currentSquareErrorCounter;
  float neuralNetwork_bestSquareError[2];
  int neuralNetwork_b_offset_squareErrorHistory;
  int neuralNetwork_b_size;
  int neuralNetwork_b_offset;
} neural_network_transform_t;

typedef struct {
  int lineOffset;
  int lineSize;
} VAL_T;

typedef struct _avg_observables {

  double energy, energy_sq, energy_error;
  double N, N_sq, N_error;
  double coulombic_energy, coulombic_energy_sq, coulombic_energy_error;
  double rd_energy, rd_energy_sq, rd_energy_error;
  double polarization_energy, polarization_energy_sq, polarization_energy_error;
  double vdw_energy, vdw_energy_sq, vdw_energy_error;
  double three_body_energy, three_body_energy_sq, three_body_energy_error;
  double dipole_rrms, dipole_rrms_sq, dipole_rrms_error;
  double density, density_sq, density_error;
  double pore_density, pore_density_error;
  double percent_wt, percent_wt_error;
  double percent_wt_me, percent_wt_me_error;
  double excess_ratio, excess_ratio_error;

  double energy_sq_sq, energy_sq_error;

  double kinetic_energy, kinetic_energy_sq, kinetic_energy_error;
  double temperature, temperature_sq, temperature_error;

  double volume, volume_sq, volume_error;

  double NU;

  double spin_ratio, spin_ratio_sq, spin_ratio_error;

  double boltzmann_factor, boltzmann_factor_sq, boltzmann_factor_error;
  double cavity_bias_probability, cavity_bias_probability_sq,
      cavity_bias_probability_error;
  double polarization_iterations, polarization_iterations_sq,
      polarization_iterations_error;

  double acceptance_rate;
  double acceptance_rate_insert, acceptance_rate_remove,
      acceptance_rate_displace;
  double acceptance_rate_adiabatic, acceptance_rate_spinflip,
      acceptance_rate_volume, acceptance_rate_ptemp;

  double qst;
  double qst_nvt;
  double heat_capacity;
  double heat_capacity_error;
  double compressibility;
  double compressibility_error;

} avg_observables_t;

typedef struct s_cone {
  float3 pos;
  float tng;
  float m1;
  float m2;
  float tex_scale;
} t_cone;

typedef struct {
  float boost;
  float overlap;
  bool active;

  float activeDutyCycle;
  float minDutyCycle;
  float overlapDutyCycle;
} Column;

typedef struct {
  unsigned int index;
  int value;
} iav_type;

typedef struct struct_char_x1 {
  char x;
} struct_char_x1;

typedef struct electrum_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[8];
  unsigned long long out[8];

} electrum_tmp_t;

typedef struct {
  Vector absoluteDir;
  float turbidity, relSize;

  Vector x, y;
  float cosThetaMax, sin2ThetaMax;
  Spectrum float3;
} SunLightParam;

typedef struct sha256_ctx_vector {
  unsigned int h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha256_ctx_vector_t;

typedef struct {
  float2 position;
  float2 velocity;
} mosquito;

typedef struct kernel_s {
  cl_program _program;
  cl_kernel _kernel;
  int _work_dim;
  size_t _globalSize[3];
  size_t _localSize[3];
} kernel_t;

typedef struct {
  int offset;
  int threshold0;
  int threshold1;
} hop;

typedef struct __attribute__((packed)) myPackedStruct {
  char2 vec;
} testPackedStruct;

typedef struct sha256_hmac_ctx_vector {
  sha256_ctx_vector_t ipad;
  sha256_ctx_vector_t opad;

} sha256_hmac_ctx_vector_t;

typedef struct {
  int index;
  float nimpulse[4];
} b2clJointImpulseNode;

typedef struct {
  float refl_r, refl_g, refl_b;
  float refrct_r, refrct_g, refrct_b;
  float ousideIor, ior;
  float R0;
  int reflectionSpecularBounce, transmitionSpecularBounce;
} GlassParam;

typedef struct sha512_ctx_vector {
  unsigned long long h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];
  unsigned int w4[4];
  unsigned int w5[4];
  unsigned int w6[4];
  unsigned int w7[4];

  int len;

} sha512_ctx_vector_t;

typedef struct LoopCtl {
  const char *ocName;
  union {
    const char *name;
    unsigned long val;
  } outBound;
  bool obConst;
  unsigned long inBound;
} LoopCtl;

typedef struct {
  uint32_t sk[32];
} des_context;

typedef struct {
  float3 pos;
  float3 normal;
  float distance;
  int objectID;
  int iter;
} TIsec;

typedef struct {

  unsigned int tileWidth, tileHeight;

  float alpha;

  float beta;

  float ss;

  float hWidth;

  float warpArea, weftArea;

  float fineness;

  float dWarpUmaxOverDWarp;
  float dWarpUmaxOverDWeft;
  float dWeftUmaxOverDWarp;
  float dWeftUmaxOverDWeft;
  float period;
} WeaveConfig;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_Rs __attribute__((aligned(4)));
  unsigned int idx_nu;
  unsigned int idx_nv;
} AshikhminSElem;

typedef struct PrivateArea {
  const char *typeName;
  unsigned int vecLen;
  unsigned int size;
} PrivateArea;

typedef struct _TriangleInterpolator {
  float2 lineA01;
  float2 lineA12;
  float2 lineA20;

  float lineC01;
  float lineC12;
  float lineC20;

  float bary01;
  float bary12;
  float bary20;

  float bary_a;
  float bary_b;
  float bary_c;

  char ext01;
  char ext12;
  char ext20;

  char draw;
} TriangleInterpolator;

typedef struct struct_char_x4 {
  char x, y, z, w;
} struct_char_x4;

typedef struct {
  union {
    int16 vecInt;
    int ints[16];
  };
} Int16CastType;

typedef struct {
  float2 wA;
  float2 wB;
  float2 w;
  float a;
  int indexA;
  int indexB;
} b2clSimplexVertex;

typedef struct {
  int distance;
} GrapheneDefaultValue;

typedef struct {
  b2clSimplexVertex m_v1, m_v2, m_v3;
  int m_count;
} b2clSimplex;

typedef struct Point2d {
  float x;
  float y;
} Point2d;

typedef struct {
  float permanence;
  float permanenceQueued;
  int targetColumn;
  uchar targetCell;
  uchar targetCellState;
} Synapse;

typedef struct __attribute__((packed)) {
  float val;
  int idx;
} ValueIndexPair;

typedef struct {
  Spectrum radiance;
  float alpha;

  float totalI;

  unsigned int largeMutationCount, smallMutationCount;
  unsigned int current, proposed, consecutiveRejects;

  float weight;
  Spectrum currentRadiance;
  float currentAlpha;
} MetropolisSampleWithAlphaChannel;

typedef struct sTrackData {
  int result;
  float error;
  float J[6];
} TrackData;

typedef struct {
  unsigned int format;

  unsigned int width;
  unsigned int height;

  unsigned int dataOffset;
} TextureMetadata;

typedef struct {
  int genes[64];
} __ShufflerChromosome;

typedef struct {
  float x, y, z;
} triple;

typedef struct _motion_estimation_desc_intel {
  unsigned int mb_block_type;
  unsigned int subpixel_mode;
  unsigned int sad_adjust_mode;
  unsigned int search_path_type;
} accelerator_intel_t;

typedef struct {
  size_t lotteryWinner;
} LocalAllocations;

typedef struct {
  local int *min;
  local int *max;
} Limits;

typedef struct {
  union {
    struct {

      float bboxMin[3];
      float bboxMax[3];
    } bvhNode;
    struct {
      unsigned int v[3];
      unsigned int meshIndex, triangleIndex;
    } triangleLeaf;
    struct {
      unsigned int leafIndex;
      unsigned int transformIndex, motionIndex;
      unsigned int meshOffsetIndex;
    } bvhLeaf;
  };

  unsigned int nodeData;
  int pad0;
} BVHArrayNode;

typedef struct {
  global Cell *cells;
  global Segment *segments;
  global Synapse *synapses;
} State;

typedef struct {
  float m[4][4];
} RT_Mat4f;

typedef struct {
  unsigned int m_key;
  unsigned int m_value;
} SortDataCL;

typedef struct {
  double fitness;
  double parameters[500];
} Creature;

typedef struct _clbpt_property {
  unsigned int root;
  int level;
} clbpt_property;

typedef struct {
  int index;
  float ks;
} GPUSpring;

typedef struct different_size_type_pair {
  long l;
  int i;
} different_size_type_pair;

typedef struct com {
  position pos;
  float mass;
} com;

typedef struct cell {
  char active;
  unsigned long depth;
  com com;
  unsigned long body_idx;
  unsigned long body_count;
  position pos;
  position size;
  unsigned long layer_idx;
} cell;

typedef struct body {
  position pos;
  position cache;
  position speed;
  float mass;
  unsigned long cell_idx;
} body;

typedef struct krb5tgs_17_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[10];
  unsigned int out[10];

} krb5tgs_17_tmp_t;

typedef struct sha512aix_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[8];
  unsigned long long out[8];

} sha512aix_tmp_t;

typedef struct {
  float p1x;
  float p2x;
  float nx;
  float ny;
  float ux;
  float uy;
  float len;
} LLine;

typedef struct def_Bounds {
  float3 dimensions;
  float3 halfDimensions;
} Bounds;

typedef struct KernelBSSRDF {
  int table_offset;
  int num_attempts;
  int pad1, pad2;
} KernelBSSRDF;

typedef struct latLong {
  float lat;
  float lng;
} LatLong;

typedef struct galaxy_infos {
  unsigned long cell_count;
  unsigned long body_count;
  unsigned long depth;
  position map_limits;
  position small_cell_size;
  unsigned long side_cell_count_lowest_level;
  float theta;
  float g;
  unsigned long max_local_work_size;
  unsigned long last_layer_idx;
  unsigned long body_buffer_offset;
  unsigned long cell_buffer_offset;
  unsigned long history_buffer_offset;
} galaxy_infos;

typedef struct {
  int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_cswap_params_t;

typedef struct {
  uint8 s0, s1, s2, s3;
} v4uint8;

typedef struct galaxy {
  cell *cells;
  body *bodies;
  galaxy_infos *infos;
} galaxy;

typedef struct edge_t_struct {

  unsigned int dst;
} edge_t;

typedef struct ocl_galaxy {
  galaxy **galaxy;

  cl_mem cells;
  cl_mem bodies;
  cl_mem sorted_bodies;
  cl_mem infos;
  cl_mem contains_losts;
  cl_mem contains_sub_dispatchables;
  cl_mem dispatch_sub_dispatchables_start_idx;
  cl_mem clear_inactive_cells_start_idx;
  cl_mem compute_com_start_idx;
  cl_mem compute_accelerations_start_idx;
  cl_mem compute_history;

  unsigned long *depth;
  unsigned long *body_count;
  unsigned long *cell_count;
  unsigned long *max_local_work_size;
  unsigned long *last_layer_idx;
  unsigned long *body_buffer_offset;
  unsigned long *cell_buffer_offset;
  unsigned long *history_buffer_offset;

  unsigned long highest_body_count;
  unsigned long highest_cell_count;
  unsigned long highest_depth;
  unsigned long highest_depth_last_layer_index;
  unsigned long galaxy_count;
  unsigned long history_size;

  float3 *quadrant_color;
  float3 *body_color;

} ocl_galaxy;

typedef struct __attribute__((aligned(4))) {
  uchar atInfinity;
  unsigned int reference __attribute__((aligned(4)));
} LightInfo;

typedef struct sha256_ctx {
  unsigned int h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha256_ctx_t;

typedef struct {
  bool isLeaf;
  float x, y, z;
  float extent;
  int start, end;
  int size;
  int childIndex[8];
} OctantLinear;

typedef struct sha224_hmac_ctx_vector {
  sha224_ctx_vector_t ipad;
  sha224_ctx_vector_t opad;

} sha224_hmac_ctx_vector_t;

typedef struct dpapimk_tmp_v1 {
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[10];
  unsigned int out[10];

  unsigned int userKey[5];

} dpapimk_tmp_v1_t;

typedef struct {
  int result;
  int depth;
} ComplexValue;

typedef struct {
  uchar data[16];
} sparse_data;

typedef struct {
  float4 Translation;
  float4 Scale;
  float4 Color;
} Instance;

typedef struct pdf {
  int V;
  int R;
  int P;

  int enc_md;

  unsigned int id_buf[8];
  unsigned int u_buf[32];
  unsigned int o_buf[32];

  int id_len;
  int o_len;
  int u_len;

  unsigned int rc4key[2];
  unsigned int rc4data[2];

} pdf_t;

typedef struct {
  double r;
  double i;
} RSComplex;

typedef struct {
  int4 s0, s1, s2, s3;
} v4int4;

typedef struct type_rect {
  float4 rightDown, leftUp;
  Color float3;
} Rectangle;

typedef struct {
  unsigned char mblim[65 + 1][32] __attribute__((aligned(32)));
  unsigned char blim[65 + 1][32] __attribute__((aligned(32)));
  unsigned char lim[65 + 1][32] __attribute__((aligned(32)));
  unsigned char hev_thr[4][32] __attribute__((aligned(32)));
  unsigned char lvl[4][4][4];
  unsigned char hev_thr_lut[2][65 + 1];
  unsigned char mode_lf_lut[10];
} loop_filter_info_n __attribute__((aligned(32)));

typedef struct {
  int id;
  char str[10];
} t_elemento;

typedef struct float10 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
} float10;

typedef struct {
  uint4 state;
  uint2 count;
  uchar buffer[64];
} MD5_CTX;

typedef struct {
  Matrix4x4 m, mInv;
} Transform;

typedef struct MotionTransform {
  Transform pre;
  Transform mid;
  Transform post;
} MotionTransform;

typedef struct KernelCamera {

  int type;

  int panorama_type;
  float fisheye_fov;
  float fisheye_lens;

  Transform cameratoworld;
  Transform rastertocamera;

  float4 dx;
  float4 dy;

  float aperturesize;
  float blades;
  float bladesrotation;
  float focaldistance;

  float shuttertime;
  int have_motion;

  float nearclip;
  float cliplength;

  float sensorwidth;
  float sensorheight;

  float width, height;
  int resolution;
  int pad1;
  int pad2;
  int pad3;

  Transform screentoworld;
  Transform rastertoworld;

  Transform worldtoscreen;
  Transform worldtoraster;
  Transform worldtondc;
  Transform worldtocamera;

  MotionTransform motion;
} KernelCamera;

typedef struct {
  ulong low, high;
} lcg12864_state;

typedef struct args_s {
  long int arg1;
} args_t;

typedef struct KernelBackground {

  int shader;
  int transparent;

  float ao_factor;
  float ao_distance;
} KernelBackground;

typedef struct {
  int size;
  int resolution;
} dt_liquify_kernel_descriptor_t;

typedef struct {
  float x;
  float y;
  float z;
} Float3;

typedef struct {
  int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_sswap_params_t;

typedef struct {
  int major;
  int minor;
  int patch;
  const char *hash;
} mkldnn_version_t;

typedef struct KernelBlackbody {
  int table_offset;
  int pad1, pad2, pad3;
} KernelBlackbody;

typedef struct KernelData {
  KernelCamera cam;
  KernelFilm film;
  KernelBackground background;
  KernelIntegrator integrator;
  KernelBVH bvh;
  KernelCurves curve;
  KernelBSSRDF bssrdf;
  KernelBlackbody blackbody;
} KernelData;

typedef struct krb5tgs_18 {
  unsigned int user[128];
  unsigned int domain[128];
  unsigned int account_info[512];
  unsigned int account_info_len;

  unsigned int checksum[3];
  unsigned int edata2[5120];
  unsigned int edata2_len;

} krb5tgs_18_t;

typedef struct {
  pbkdf2_salt pbkdf2;
  uchar data[4];
  int length;
} blockchain_salt;

typedef struct HiddenLayer {
  int numberOfNodes;
  Node nodes[3400];
} HiddenLayer;

typedef struct {
  void *data;
  bool isInitialized;
} field_t;

typedef struct {
  salt_t pbkdf2;
  unsigned int key_wrapping_rounds;
  uchar salt[64];
  uchar wrappedkey[144];
} axcrypt2_salt_t;

typedef struct {
  __constant int *int_pointer;
  __constant char *char_pointer;
  __constant float *float_pointer;
  __constant float4 *vector_pointer;
} ConstantPrimitivePointerStruct;

typedef struct {
  int i;
  float f;
} ArrayStruct;

typedef struct {
  unsigned char op;
  unsigned char bits;
  unsigned short val;
} code;

typedef struct {
  unsigned int cracked;
  unsigned int key[16 / 4];
} agile_out;

typedef struct {
  float4 z;
  float iters;
  float distance;
  float colorIndex;
  float orbitTrapR;
  int objectId;
  bool maxiter;
} formulaOut;

typedef struct {
  __local int (*int_array_pointer)[1];
  __local char (*char_array_pointer)[1];
  __local float (*float_array_pointer)[1];
  __local float4 (*vector_array_pointer)[1];
} LocalArrayPointerStruct;

typedef struct {
  PrimitiveStruct primitive_struct;
  __local PrimitiveStruct *primitive_struct_pointer;
  PrimitiveStruct primitive_struct_array[1];
  LocalPrimitivePointerStruct primitive_pointer_struct;
  __local LocalPrimitivePointerStruct *primitive_pointer_struct_pointer;
  LocalPrimitivePointerStruct primitive_pointer_struct_array[1];
  ArrayStruct array_struct;
  __local ArrayStruct *array_struct_pointer;
  ArrayStruct array_struct_array[1];
  LocalArrayPointerStruct array_pointer_struct;
  __local LocalArrayPointerStruct *array_pointer_struct_pointer;
  LocalArrayPointerStruct array_pointer_struct_array[1];
} LocalMainStruct;

typedef struct __attribute__((aligned(4))) {
  uchar texType;
  unsigned int width __attribute__((aligned(4))), height;
  int offsetData;
} ImageTexture;

typedef struct {

  uint64_t flags;

  int compensation_mask;

  float scale_adjust;

  char reserved[64];
} mkldnn_memory_extra_desc_t;

typedef struct {
  DPTYPE_VEC data[(416 / 32)];
} DPTYPE_PE_VEC;

typedef struct {
  uint32_t size[1024];
  uint64_t d[20 * 1024];
} mpzcll_t;

typedef struct {
  float mutr;
  float mua;
  float g;
  float n;

  float flc;
  float muaf;
  float eY;
  float albedof;
} BulkStruct;

typedef struct {
  __constant float *constant_float_ptr;
} ConstantArrayPointerStruct;

typedef struct {
  PrimitiveStruct primitive_struct;
  __constant PrimitiveStruct *primitive_struct_pointer;
  PrimitiveStruct primitive_struct_array[1];
  ConstantPrimitivePointerStruct primitive_pointer_struct;
  __constant ConstantPrimitivePointerStruct *primitive_pointer_struct_pointer;
  ConstantPrimitivePointerStruct primitive_pointer_struct_array[1];
  ArrayStruct array_struct;
  __constant ArrayStruct *array_struct_pointer;
  ArrayStruct array_struct_array[1];
  ConstantArrayPointerStruct array_pointer_struct;
  __constant ConstantArrayPointerStruct *array_pointer_struct_pointer;
  ConstantArrayPointerStruct array_pointer_struct_array[1];
} ConstantMainStruct;

typedef struct blake2 {
  unsigned long long h[8];
  unsigned long long t[2];
  unsigned long long f[2];
  unsigned int buflen;
  unsigned int outlen;
  unsigned char last_node;

} blake2_t;

typedef struct sha1aix_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[5];
  unsigned int out[5];

} sha1aix_tmp_t;

typedef struct HitInfo {

  int triangle_ID, light_ID;
  float4 hit_point;
  float4 normal;

} HitInfo;

typedef struct {
  const char *data;
  size_t size;
} TVMByteArray;

typedef struct record {
  int value;
} record;

typedef struct {
  float2 e00;
  float2 e01;
  float2 e02;
  float2 e10;
  float2 e11;
  float2 e12;
  float2 e20;
  float2 e21;
  float2 e22;
} Matrixsu3;

typedef struct {
  float2 pointA;
  float2 pointB;
  float ndistance;
  int iterations;
} b2clDistanceOutput;

typedef struct axcrypt2 {
  unsigned int salt[16];
  unsigned int data[36];

} axcrypt2_t;

typedef struct __attribute__((packed)) _chan0_t {
  uchar ctrl;
  stack_t s;
} chan0_t;

typedef struct QuadRay {
  float4 ox, oy, oz;
  float4 dx, dy, dz;
  float4 mint, maxt;
} QuadRay;

typedef struct b2clGearJointData {
  int joint1;
  int joint2;
  int typeA;
  int typeB;

  float localAnchorA[2];
  float localAnchorB[2];
  float localAnchorC[2];
  float localAnchorD[2];

  float localAxisC[2];
  float localAxisD[2];

  float referenceAngleA;
  float referenceAngleB;

  float gearConstant;
  float ratio;

  float lcA[2], lcB[2], lcC[2], lcD[2];
  float mA, mB, mC, mD;
  float iA, iB, iC, iD;
  float JvAC[2], JvBD[2];
  float JwA, JwB, JwC, JwD;
  float mass;
} b2clGearJointData;

typedef struct {
  local int *min;
  local int *max;
} Inner;

typedef struct {
  int a_field_identifier_that_is_255_characters_long_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_12345678;
} ATypeIdentifierThatIs255CharactersLong0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456;

typedef struct {
  float4 m_plane;
  int m_indexOffset;
  int m_numIndices;
} b3GpuFace;

typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray *scale;
  TfLiteIntArray *zero_point;
  int quantized_dimension;
} TfLiteAffineQuantization;

typedef struct {
  int a;
  int b[1];
} LocalStruct;

typedef struct {

  uchar4

      pixels[144];
  int xsize;
  int ysize;
  int zsize;
  int padding;
} astc_codec_image;

typedef struct subt_rec {
  unsigned int service_id;
  unsigned long long args[16];
  uchar arg_status[16];
  uchar subt_status;
  uchar return_to;
  ushort return_as;
} subt_rec;

typedef struct wpaaes_tmp {
  unsigned int ipad[8];
  unsigned int opad[8];

  unsigned int dgst[8];
  unsigned int out[8];

} wpaaes_tmp_t;

typedef struct ccl_wrapper_info {

  void *value;

  size_t size;

} CCLWrapperInfo;

typedef struct {
  Spectrum radiance;
} RandomSampleWithoutAlphaChannel;

typedef struct {
  unsigned int interpolatedTransformFirstIndex;
  unsigned int interpolatedTransformLastIndex;
} MotionSystem;

typedef struct struct_arr32 {
  int arr[32];
} struct_arr32;

typedef struct Bucket Bucket;

constant uchar8 perm_rd[256] = {
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(1, 2, 3, 4, 5, 6, 7, 0),
    (uchar8)(1, 2, 3, 4, 5, 6, 0, 7), (uchar8)(2, 3, 4, 5, 6, 7, 0, 1),
    (uchar8)(1, 2, 3, 4, 5, 0, 6, 7), (uchar8)(2, 3, 4, 5, 6, 0, 7, 1),
    (uchar8)(2, 3, 4, 5, 6, 0, 1, 7), (uchar8)(3, 4, 5, 6, 7, 0, 1, 2),
    (uchar8)(1, 2, 3, 4, 0, 5, 6, 7), (uchar8)(2, 3, 4, 5, 0, 6, 7, 1),
    (uchar8)(2, 3, 4, 5, 0, 6, 1, 7), (uchar8)(3, 4, 5, 6, 0, 7, 1, 2),
    (uchar8)(2, 3, 4, 5, 0, 1, 6, 7), (uchar8)(3, 4, 5, 6, 0, 1, 7, 2),
    (uchar8)(3, 4, 5, 6, 0, 1, 2, 7), (uchar8)(4, 5, 6, 7, 0, 1, 2, 3),
    (uchar8)(1, 2, 3, 0, 4, 5, 6, 7), (uchar8)(2, 3, 4, 0, 5, 6, 7, 1),
    (uchar8)(2, 3, 4, 0, 5, 6, 1, 7), (uchar8)(3, 4, 5, 0, 6, 7, 1, 2),
    (uchar8)(2, 3, 4, 0, 5, 1, 6, 7), (uchar8)(3, 4, 5, 0, 6, 1, 7, 2),
    (uchar8)(3, 4, 5, 0, 6, 1, 2, 7), (uchar8)(4, 5, 6, 0, 7, 1, 2, 3),
    (uchar8)(2, 3, 4, 0, 1, 5, 6, 7), (uchar8)(3, 4, 5, 0, 1, 6, 7, 2),
    (uchar8)(3, 4, 5, 0, 1, 6, 2, 7), (uchar8)(4, 5, 6, 0, 1, 7, 2, 3),
    (uchar8)(3, 4, 5, 0, 1, 2, 6, 7), (uchar8)(4, 5, 6, 0, 1, 2, 7, 3),
    (uchar8)(4, 5, 6, 0, 1, 2, 3, 7), (uchar8)(5, 6, 7, 0, 1, 2, 3, 4),
    (uchar8)(1, 2, 0, 3, 4, 5, 6, 7), (uchar8)(2, 3, 0, 4, 5, 6, 7, 1),
    (uchar8)(2, 3, 0, 4, 5, 6, 1, 7), (uchar8)(3, 4, 0, 5, 6, 7, 1, 2),
    (uchar8)(2, 3, 0, 4, 5, 1, 6, 7), (uchar8)(3, 4, 0, 5, 6, 1, 7, 2),
    (uchar8)(3, 4, 0, 5, 6, 1, 2, 7), (uchar8)(4, 5, 0, 6, 7, 1, 2, 3),
    (uchar8)(2, 3, 0, 4, 1, 5, 6, 7), (uchar8)(3, 4, 0, 5, 1, 6, 7, 2),
    (uchar8)(3, 4, 0, 5, 1, 6, 2, 7), (uchar8)(4, 5, 0, 6, 1, 7, 2, 3),
    (uchar8)(3, 4, 0, 5, 1, 2, 6, 7), (uchar8)(4, 5, 0, 6, 1, 2, 7, 3),
    (uchar8)(4, 5, 0, 6, 1, 2, 3, 7), (uchar8)(5, 6, 0, 7, 1, 2, 3, 4),
    (uchar8)(2, 3, 0, 1, 4, 5, 6, 7), (uchar8)(3, 4, 0, 1, 5, 6, 7, 2),
    (uchar8)(3, 4, 0, 1, 5, 6, 2, 7), (uchar8)(4, 5, 0, 1, 6, 7, 2, 3),
    (uchar8)(3, 4, 0, 1, 5, 2, 6, 7), (uchar8)(4, 5, 0, 1, 6, 2, 7, 3),
    (uchar8)(4, 5, 0, 1, 6, 2, 3, 7), (uchar8)(5, 6, 0, 1, 7, 2, 3, 4),
    (uchar8)(3, 4, 0, 1, 2, 5, 6, 7), (uchar8)(4, 5, 0, 1, 2, 6, 7, 3),
    (uchar8)(4, 5, 0, 1, 2, 6, 3, 7), (uchar8)(5, 6, 0, 1, 2, 7, 3, 4),
    (uchar8)(4, 5, 0, 1, 2, 3, 6, 7), (uchar8)(5, 6, 0, 1, 2, 3, 7, 4),
    (uchar8)(5, 6, 0, 1, 2, 3, 4, 7), (uchar8)(6, 7, 0, 1, 2, 3, 4, 5),
    (uchar8)(1, 0, 2, 3, 4, 5, 6, 7), (uchar8)(2, 0, 3, 4, 5, 6, 7, 1),
    (uchar8)(2, 0, 3, 4, 5, 6, 1, 7), (uchar8)(3, 0, 4, 5, 6, 7, 1, 2),
    (uchar8)(2, 0, 3, 4, 5, 1, 6, 7), (uchar8)(3, 0, 4, 5, 6, 1, 7, 2),
    (uchar8)(3, 0, 4, 5, 6, 1, 2, 7), (uchar8)(4, 0, 5, 6, 7, 1, 2, 3),
    (uchar8)(2, 0, 3, 4, 1, 5, 6, 7), (uchar8)(3, 0, 4, 5, 1, 6, 7, 2),
    (uchar8)(3, 0, 4, 5, 1, 6, 2, 7), (uchar8)(4, 0, 5, 6, 1, 7, 2, 3),
    (uchar8)(3, 0, 4, 5, 1, 2, 6, 7), (uchar8)(4, 0, 5, 6, 1, 2, 7, 3),
    (uchar8)(4, 0, 5, 6, 1, 2, 3, 7), (uchar8)(5, 0, 6, 7, 1, 2, 3, 4),
    (uchar8)(2, 0, 3, 1, 4, 5, 6, 7), (uchar8)(3, 0, 4, 1, 5, 6, 7, 2),
    (uchar8)(3, 0, 4, 1, 5, 6, 2, 7), (uchar8)(4, 0, 5, 1, 6, 7, 2, 3),
    (uchar8)(3, 0, 4, 1, 5, 2, 6, 7), (uchar8)(4, 0, 5, 1, 6, 2, 7, 3),
    (uchar8)(4, 0, 5, 1, 6, 2, 3, 7), (uchar8)(5, 0, 6, 1, 7, 2, 3, 4),
    (uchar8)(3, 0, 4, 1, 2, 5, 6, 7), (uchar8)(4, 0, 5, 1, 2, 6, 7, 3),
    (uchar8)(4, 0, 5, 1, 2, 6, 3, 7), (uchar8)(5, 0, 6, 1, 2, 7, 3, 4),
    (uchar8)(4, 0, 5, 1, 2, 3, 6, 7), (uchar8)(5, 0, 6, 1, 2, 3, 7, 4),
    (uchar8)(5, 0, 6, 1, 2, 3, 4, 7), (uchar8)(6, 0, 7, 1, 2, 3, 4, 5),
    (uchar8)(2, 0, 1, 3, 4, 5, 6, 7), (uchar8)(3, 0, 1, 4, 5, 6, 7, 2),
    (uchar8)(3, 0, 1, 4, 5, 6, 2, 7), (uchar8)(4, 0, 1, 5, 6, 7, 2, 3),
    (uchar8)(3, 0, 1, 4, 5, 2, 6, 7), (uchar8)(4, 0, 1, 5, 6, 2, 7, 3),
    (uchar8)(4, 0, 1, 5, 6, 2, 3, 7), (uchar8)(5, 0, 1, 6, 7, 2, 3, 4),
    (uchar8)(3, 0, 1, 4, 2, 5, 6, 7), (uchar8)(4, 0, 1, 5, 2, 6, 7, 3),
    (uchar8)(4, 0, 1, 5, 2, 6, 3, 7), (uchar8)(5, 0, 1, 6, 2, 7, 3, 4),
    (uchar8)(4, 0, 1, 5, 2, 3, 6, 7), (uchar8)(5, 0, 1, 6, 2, 3, 7, 4),
    (uchar8)(5, 0, 1, 6, 2, 3, 4, 7), (uchar8)(6, 0, 1, 7, 2, 3, 4, 5),
    (uchar8)(3, 0, 1, 2, 4, 5, 6, 7), (uchar8)(4, 0, 1, 2, 5, 6, 7, 3),
    (uchar8)(4, 0, 1, 2, 5, 6, 3, 7), (uchar8)(5, 0, 1, 2, 6, 7, 3, 4),
    (uchar8)(4, 0, 1, 2, 5, 3, 6, 7), (uchar8)(5, 0, 1, 2, 6, 3, 7, 4),
    (uchar8)(5, 0, 1, 2, 6, 3, 4, 7), (uchar8)(6, 0, 1, 2, 7, 3, 4, 5),
    (uchar8)(4, 0, 1, 2, 3, 5, 6, 7), (uchar8)(5, 0, 1, 2, 3, 6, 7, 4),
    (uchar8)(5, 0, 1, 2, 3, 6, 4, 7), (uchar8)(6, 0, 1, 2, 3, 7, 4, 5),
    (uchar8)(5, 0, 1, 2, 3, 4, 6, 7), (uchar8)(6, 0, 1, 2, 3, 4, 7, 5),
    (uchar8)(6, 0, 1, 2, 3, 4, 5, 7), (uchar8)(7, 0, 1, 2, 3, 4, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 2, 3, 4, 5, 6, 7, 1),
    (uchar8)(0, 2, 3, 4, 5, 6, 1, 7), (uchar8)(0, 3, 4, 5, 6, 7, 1, 2),
    (uchar8)(0, 2, 3, 4, 5, 1, 6, 7), (uchar8)(0, 3, 4, 5, 6, 1, 7, 2),
    (uchar8)(0, 3, 4, 5, 6, 1, 2, 7), (uchar8)(0, 4, 5, 6, 7, 1, 2, 3),
    (uchar8)(0, 2, 3, 4, 1, 5, 6, 7), (uchar8)(0, 3, 4, 5, 1, 6, 7, 2),
    (uchar8)(0, 3, 4, 5, 1, 6, 2, 7), (uchar8)(0, 4, 5, 6, 1, 7, 2, 3),
    (uchar8)(0, 3, 4, 5, 1, 2, 6, 7), (uchar8)(0, 4, 5, 6, 1, 2, 7, 3),
    (uchar8)(0, 4, 5, 6, 1, 2, 3, 7), (uchar8)(0, 5, 6, 7, 1, 2, 3, 4),
    (uchar8)(0, 2, 3, 1, 4, 5, 6, 7), (uchar8)(0, 3, 4, 1, 5, 6, 7, 2),
    (uchar8)(0, 3, 4, 1, 5, 6, 2, 7), (uchar8)(0, 4, 5, 1, 6, 7, 2, 3),
    (uchar8)(0, 3, 4, 1, 5, 2, 6, 7), (uchar8)(0, 4, 5, 1, 6, 2, 7, 3),
    (uchar8)(0, 4, 5, 1, 6, 2, 3, 7), (uchar8)(0, 5, 6, 1, 7, 2, 3, 4),
    (uchar8)(0, 3, 4, 1, 2, 5, 6, 7), (uchar8)(0, 4, 5, 1, 2, 6, 7, 3),
    (uchar8)(0, 4, 5, 1, 2, 6, 3, 7), (uchar8)(0, 5, 6, 1, 2, 7, 3, 4),
    (uchar8)(0, 4, 5, 1, 2, 3, 6, 7), (uchar8)(0, 5, 6, 1, 2, 3, 7, 4),
    (uchar8)(0, 5, 6, 1, 2, 3, 4, 7), (uchar8)(0, 6, 7, 1, 2, 3, 4, 5),
    (uchar8)(0, 2, 1, 3, 4, 5, 6, 7), (uchar8)(0, 3, 1, 4, 5, 6, 7, 2),
    (uchar8)(0, 3, 1, 4, 5, 6, 2, 7), (uchar8)(0, 4, 1, 5, 6, 7, 2, 3),
    (uchar8)(0, 3, 1, 4, 5, 2, 6, 7), (uchar8)(0, 4, 1, 5, 6, 2, 7, 3),
    (uchar8)(0, 4, 1, 5, 6, 2, 3, 7), (uchar8)(0, 5, 1, 6, 7, 2, 3, 4),
    (uchar8)(0, 3, 1, 4, 2, 5, 6, 7), (uchar8)(0, 4, 1, 5, 2, 6, 7, 3),
    (uchar8)(0, 4, 1, 5, 2, 6, 3, 7), (uchar8)(0, 5, 1, 6, 2, 7, 3, 4),
    (uchar8)(0, 4, 1, 5, 2, 3, 6, 7), (uchar8)(0, 5, 1, 6, 2, 3, 7, 4),
    (uchar8)(0, 5, 1, 6, 2, 3, 4, 7), (uchar8)(0, 6, 1, 7, 2, 3, 4, 5),
    (uchar8)(0, 3, 1, 2, 4, 5, 6, 7), (uchar8)(0, 4, 1, 2, 5, 6, 7, 3),
    (uchar8)(0, 4, 1, 2, 5, 6, 3, 7), (uchar8)(0, 5, 1, 2, 6, 7, 3, 4),
    (uchar8)(0, 4, 1, 2, 5, 3, 6, 7), (uchar8)(0, 5, 1, 2, 6, 3, 7, 4),
    (uchar8)(0, 5, 1, 2, 6, 3, 4, 7), (uchar8)(0, 6, 1, 2, 7, 3, 4, 5),
    (uchar8)(0, 4, 1, 2, 3, 5, 6, 7), (uchar8)(0, 5, 1, 2, 3, 6, 7, 4),
    (uchar8)(0, 5, 1, 2, 3, 6, 4, 7), (uchar8)(0, 6, 1, 2, 3, 7, 4, 5),
    (uchar8)(0, 5, 1, 2, 3, 4, 6, 7), (uchar8)(0, 6, 1, 2, 3, 4, 7, 5),
    (uchar8)(0, 6, 1, 2, 3, 4, 5, 7), (uchar8)(0, 7, 1, 2, 3, 4, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 3, 4, 5, 6, 7, 2),
    (uchar8)(0, 1, 3, 4, 5, 6, 2, 7), (uchar8)(0, 1, 4, 5, 6, 7, 2, 3),
    (uchar8)(0, 1, 3, 4, 5, 2, 6, 7), (uchar8)(0, 1, 4, 5, 6, 2, 7, 3),
    (uchar8)(0, 1, 4, 5, 6, 2, 3, 7), (uchar8)(0, 1, 5, 6, 7, 2, 3, 4),
    (uchar8)(0, 1, 3, 4, 2, 5, 6, 7), (uchar8)(0, 1, 4, 5, 2, 6, 7, 3),
    (uchar8)(0, 1, 4, 5, 2, 6, 3, 7), (uchar8)(0, 1, 5, 6, 2, 7, 3, 4),
    (uchar8)(0, 1, 4, 5, 2, 3, 6, 7), (uchar8)(0, 1, 5, 6, 2, 3, 7, 4),
    (uchar8)(0, 1, 5, 6, 2, 3, 4, 7), (uchar8)(0, 1, 6, 7, 2, 3, 4, 5),
    (uchar8)(0, 1, 3, 2, 4, 5, 6, 7), (uchar8)(0, 1, 4, 2, 5, 6, 7, 3),
    (uchar8)(0, 1, 4, 2, 5, 6, 3, 7), (uchar8)(0, 1, 5, 2, 6, 7, 3, 4),
    (uchar8)(0, 1, 4, 2, 5, 3, 6, 7), (uchar8)(0, 1, 5, 2, 6, 3, 7, 4),
    (uchar8)(0, 1, 5, 2, 6, 3, 4, 7), (uchar8)(0, 1, 6, 2, 7, 3, 4, 5),
    (uchar8)(0, 1, 4, 2, 3, 5, 6, 7), (uchar8)(0, 1, 5, 2, 3, 6, 7, 4),
    (uchar8)(0, 1, 5, 2, 3, 6, 4, 7), (uchar8)(0, 1, 6, 2, 3, 7, 4, 5),
    (uchar8)(0, 1, 5, 2, 3, 4, 6, 7), (uchar8)(0, 1, 6, 2, 3, 4, 7, 5),
    (uchar8)(0, 1, 6, 2, 3, 4, 5, 7), (uchar8)(0, 1, 7, 2, 3, 4, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 2, 4, 5, 6, 7, 3),
    (uchar8)(0, 1, 2, 4, 5, 6, 3, 7), (uchar8)(0, 1, 2, 5, 6, 7, 3, 4),
    (uchar8)(0, 1, 2, 4, 5, 3, 6, 7), (uchar8)(0, 1, 2, 5, 6, 3, 7, 4),
    (uchar8)(0, 1, 2, 5, 6, 3, 4, 7), (uchar8)(0, 1, 2, 6, 7, 3, 4, 5),
    (uchar8)(0, 1, 2, 4, 3, 5, 6, 7), (uchar8)(0, 1, 2, 5, 3, 6, 7, 4),
    (uchar8)(0, 1, 2, 5, 3, 6, 4, 7), (uchar8)(0, 1, 2, 6, 3, 7, 4, 5),
    (uchar8)(0, 1, 2, 5, 3, 4, 6, 7), (uchar8)(0, 1, 2, 6, 3, 4, 7, 5),
    (uchar8)(0, 1, 2, 6, 3, 4, 5, 7), (uchar8)(0, 1, 2, 7, 3, 4, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 2, 3, 5, 6, 7, 4),
    (uchar8)(0, 1, 2, 3, 5, 6, 4, 7), (uchar8)(0, 1, 2, 3, 6, 7, 4, 5),
    (uchar8)(0, 1, 2, 3, 5, 4, 6, 7), (uchar8)(0, 1, 2, 3, 6, 4, 7, 5),
    (uchar8)(0, 1, 2, 3, 6, 4, 5, 7), (uchar8)(0, 1, 2, 3, 7, 4, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 2, 3, 4, 6, 7, 5),
    (uchar8)(0, 1, 2, 3, 4, 6, 5, 7), (uchar8)(0, 1, 2, 3, 4, 7, 5, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 2, 3, 4, 5, 7, 6),
    (uchar8)(0, 1, 2, 3, 4, 5, 6, 7), (uchar8)(0, 1, 2, 3, 4, 5, 6, 7)};

typedef struct PVPatch_ {
  unsigned int offset;
  unsigned short nx, ny;

  float padding;

} PVPatch __attribute__((aligned));

typedef struct {
  unsigned int tl;
  unsigned int t;
  unsigned int tr;

  unsigned int l;
  unsigned int c;
  unsigned int r;

  unsigned int bl;
  unsigned int b;
  unsigned int br;

  short pos;
} CtxWindow;

typedef struct {
  unsigned int prefix_scan_kernel__scalar_histogram[20];
  uint4 prefix_scan_128__vector_histogram;
  unsigned int prefix_scan_kernel__unused_top_level_prefix_sum;
} WclPrivates;

typedef struct bc7_enc_state {
  float opaque_err;
  float best_err;
  unsigned int best_data[5];
} bc7_enc_state;

typedef struct bsdicrypt_tmp {
  unsigned int Kc[16];
  unsigned int Kd[16];

  unsigned int iv[2];

} bsdicrypt_tmp_t;

typedef struct {
  uint32_t size[1024];
  uint64_t d[40 * 1024];
} mpzclel_t;

typedef struct pdf17l8_tmp {
  union {
    unsigned int dgst32[16];
    unsigned long long dgst64[8];
  };

  unsigned int dgst_len;
  unsigned int W_len;

} pdf17l8_tmp_t;

typedef struct {
  float refl_r, refl_g, refl_b;
  float refrct_r, refrct_g, refrct_b;
  float transFilter, totFilter, reflPdf, transPdf;
  bool reflectionSpecularBounce, transmitionSpecularBounce;
} ArchGlassParam;

typedef struct dt_iop_roi_t {
  int x, y, width, height;
  float scale;
} dt_iop_roi_t;

typedef struct {
  uint32_t h[8];
  uint32_t t;
} state256;

typedef struct krb5tgs_18_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[16];
  unsigned int out[16];

} krb5tgs_18_tmp_t;

typedef struct {
  float points[3];
  float vector[3];
  float force;
} EXTOBJ_force_t;

typedef struct {
  unsigned int length;
  uchar pass[256];
} keystore_password;

typedef struct {
  int forces_num;
  EXTOBJ_force_t forces[100];
} force_pack_t, *force_pack_p;

typedef struct {
  __local LocalAllocations *localAllocations_min, *localAllocations_max;
  __local float4 *scratch_min, *scratch_max;
} LocalLimits;

typedef struct {
  size_t awesomize__gid;
  size_t awesomize__wgid;
  size_t awesomize__wgsize;
  float4 flip_to_awesomeness__index_vec;
  float flip_to_awesomeness__index_float;
  int flip_to_awesomeness__index;
} PrivateAllocations;

typedef struct _PAPhysicsFalloffData {
  float strength, max, min;
} PAPhysicsFalloffData;

typedef struct {
  float x, y, z;
  float q;
  float fx, fy, fz;
  float padding;

} AtomData;

typedef struct {
  ConstantLimits constantLimits;
  GlobalLimits globalLimits;
  LocalLimits localLimits;

  PrivateAllocations privateAllocations;
} ProgramAllocations;

typedef struct {
  unsigned int numGeneratedTripcodes;
  unsigned char numMatchingTripcodes;
  TripcodeKeyPair pair;
} GPUOutput;

typedef struct _PAPhysicsTurbulenceData {
  float strength, size, noise;
  PAPhysicsFalloffData falloff;
} PAPhysicsTurbulenceData;

typedef struct _UberV2ShaderData {
  float4 diffuse_color;
  float4 reflection_color;
  float4 coating_color;
  float4 refraction_color;
  float4 emission_color;
  float4 sss_absorption_color;
  float4 sss_scatter_color;
  float4 sss_subsurface_color;
  float4 shading_normal;

  float reflection_roughness;
  float reflection_anisotropy;
  float reflection_anisotropy_rotation;
  float reflection_ior;

  float reflection_metalness;
  float coating_ior;
  float refraction_roughness;
  float refraction_ior;

  float transparency;
  float sss_absorption_distance;
  float sss_scatter_distance;
  float sss_scatter_direction;

} UberV2ShaderData;

typedef struct shifts shifts;
struct shifts {
  struct shifts *next;
  short number;
  short nshifts;
  short shift[1];
};

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_M __attribute__((aligned(4)));
} DiffuseLElem;

typedef struct Collidable {
  int m_unused1;
  int m_unused2;
  int m_shapeType;
  int m_shapeIndex;
} Collidable;

typedef struct {
  uint32_t hash[8 * 1024];
  uint32_t hashTemp[20 * 1024];
  mpzcl_t mpzPrimeMin;
} shaTemp_t;

typedef struct {
  salt_t pbkdf2;
  uint32_t md_version;
  uint16_t md_ealgo;
  uint16_t md_keylen;
  uint16_t md_aalgo;
  uint8_t md_keys;
  uint8_t md_mkeys[2 * ((64 + 64) + 64)];
} geli_salt_t;

typedef struct _NclFileVarInfo {
  char fvarname[256];
  int level;
  unsigned int offset;
  struct _NclSymbol *parent_file;
} NclFileVarInfo;

typedef struct {

  long start;

  long end;
} IndexRange;

typedef struct unz_global_info_s {
  ulong number_entry;

  ulong size_comment;
} unz_global_info;

typedef struct {
  Spectrum float3;
  Vector absoluteLightDir;
} SharpDistantLightParam;

typedef struct {
  int s0, s1, s2, s3;
} v4int;

typedef struct axcrypt2_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[8];
  unsigned long long out[8];

  unsigned int KEK[8];
  unsigned int data[14];

} axcrypt2_tmp_t;

typedef struct Quality {
  float dd;
  int dPrec;
  int ao_nb;
  float ao_dd;
  float ao_dmax;
  int sh_on;
  float sh_dd;
  float sh_dmax;
} Quality;

typedef struct s_parallelogram {
  float3 pos;
  float h;
  float w;
  float l;
  float tex_scale;
} t_parallelogram;

typedef struct {

  float3 point;

  float3 normal;

  float2 uv;

  unsigned int matNodeIndex;
} Surface;

typedef struct Partitioner {

  int n_tasks;
  int cut;
  int current;
} Partitioner;

typedef struct {
  int type;
  float radius;
  Vector center, light, normal;
  Surface surface;
} Item;

typedef struct mkldnn_primitive_desc *mkldnn_primitive_desc_t;
typedef enum {

  mkldnn_scratchpad_mode_library,

  mkldnn_scratchpad_mode_user,
} mkldnn_scratchpad_mode_t;

typedef struct NoExpressao {
  char expr[128 * 10];
  struct NoExpressao *proximo;
} No;

typedef struct pbkdf2_md5 {
  unsigned int salt_buf[16];

} pbkdf2_md5_t;

typedef struct __attribute__((aligned(4))) {

  float uDir[2];
} IDFSample;

typedef struct matrix3 {
  float3 m[3];
} matrix3;

typedef struct {
  unsigned long int state[8];
} sha256_context;

typedef struct lastpass_tmp {
  unsigned int ipad[8];
  unsigned int opad[8];

  unsigned int dgst[8];
  unsigned int out[8];

} lastpass_tmp_t;

typedef struct wpa_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} wpa_tmp_t;

typedef struct {
  cl_uint group_id;
  cl_char isnull;
  cl_char __padding__[3];
  union {
    cl_short short_val;
    cl_int int_val;
    cl_long long_val;
    float float_val;
    cl_double double_val;
  };
} pagg_datum;

typedef struct {
  int2 start_index;
  int2 end_index;
  float2 start_continuous_index;
  float2 end_continuous_index;
} GPUImageFunction2D;

typedef struct CameraData {
  float3 viewDir;
  float3 rightDir;
  float3 upDir;
  float3 origin;
} CameraData;

typedef struct {
  float4 bbMin;
  float4 bbMax;
  unsigned int left;
  unsigned int right;
  unsigned int bit;
  unsigned int trav;
} BVHNode;

typedef struct {
  double real, imag;
} bh_complex128;

typedef struct {
  float a0, ax, ay, ahalfxx, ahalfyy, axy;
} QuadraticCoeffs;

typedef struct sort_info {

  float route_length;

  __global const unsigned int *idx;
} sort_t;

typedef struct _NclRefList {

  int pid;
  struct _NclRefList *next;
} NclRefList;

typedef struct {
  int i1, i2;

  float2 v1, v2;

  float2 normal;

  float2 sideNormal1;
  float sideOffset1;

  float2 sideNormal2;
  float sideOffset2;
} b2clReferenceFace;

typedef struct netntlm {
  unsigned int user_len;
  unsigned int domain_len;
  unsigned int srvchall_len;
  unsigned int clichall_len;

  unsigned int userdomain_buf[64];
  unsigned int chall_buf[256];

} netntlm_t;

typedef struct {
  int parent;
  int left;
  int right;
  int next;
} HlbvhNode;

typedef struct {
  int n, lda, j0, npivots;
  short ipiv[32];
} zlaswp_params_t2;

typedef struct _peng_robinson_constants {
  double Tc;
  double Pc;
  double w;
} peng_robinson_constants;

typedef struct {
  int type;
  int pointCount;
  float localPointX;
  float localPointY;
  float localNormalX;
  float localNormalY;
  float point0X;
  float point0Y;
  float point1X;
  float point1Y;
} clb2SDManifold;

typedef struct {

  unsigned int type;

  unsigned int leftChild;

  union {
    unsigned int rightChild;

    int transmittanceTex;
  };

  union {

    int bumpTex;
    int mixWeightsTex;

    int reflectanceTex;
    int specularityTex;
    int radianceTex;
  };

  union {
    float3 reflectance;
    float3 specularity;
    float3 radiance;
    float3 intDispersionIORs;

    float mixWeight;
  };

  union {
    float3 transmittance;
    float3 extDispersionIORs;
  };

  union {
    float intIOR;
  };

  union {
    float extIOR;
  };

  union {

    float scale;

    float roughness;
  };

  union {
    int roughnessTex;
  };
} MaterialNode;

typedef struct {
  Float3 colliderPos;
  float colliderRadius;

  Float3 gravity;
  float globalDamping;
  float particleRadius;

  uint3 gridSize;
  unsigned int numCells;
  Float3 worldOrigin;
  Float3 cellSize;

  unsigned int numBodies;
  unsigned int maxParticlesPerCell;

  float spring;
  float damping;
  float shear;
  float attraction;
  float boundaryDamping;
} simParams_t;

typedef struct Mat4x4 {
  float4 r1;
  float4 r2;
  float4 r3;
  float4 r4;
} Mat4x4;

typedef struct {

  float4 transformMat0;
  float4 transformMat1;
  float4 transformMat2;
  float4 transformMat3;

  float area;

  unsigned int triIndex;

  unsigned int matNodeIndex;

  unsigned int type;
} Emissive;

typedef struct {
  unsigned int rr[(1 << (8))];
  unsigned int mm[(1 << (8))];
  unsigned int aa;
  unsigned int bb;
  unsigned int cc;
  unsigned int idx;
} isaac_state;

typedef struct {
  unsigned int kdTexIndex;
  unsigned int p1TexIndex;
  unsigned int p2TexIndex;
  unsigned int p3TexIndex;
  unsigned int thicknessTexIndex;
} VelvetParam;

typedef struct {
  __global float *pH;
  __global float *pU;
  __global float *pV;
} CFluxVecPtr;

typedef struct __attribute__((aligned(1))) {
  uchar _type;
} DistributionHead;

typedef struct __attribute__((aligned(16))) {
  float3 min;
  float3 max;
  bool isChild[2];
  unsigned int c[2];
} InternalNode;

typedef struct {
  uint32_t buffer[32];
  uint32_t buflen;
} xsha512_ctx;

typedef struct __attribute__((aligned(4))) {
  DistributionHead head;
  unsigned int numValues __attribute__((aligned(4)));
  float startDomain, endDomain;
  float widthStratum;
  int offsetPDF;
  int offsetCDF;
} ContinuousConsts1D;

typedef struct {
  int a;
} Struct;

typedef struct {
  float w[512];
} lmb3;

typedef struct __attribute__((aligned(4))) {
  DistributionHead head;
  int offsetChildren __attribute__((aligned(4)));
  ContinuousConsts1D distParent;
} ContinuousConsts2D_H;

typedef struct ge25519_t {
  unsigned int x, y, z, t;
} ge25519;

typedef struct ray_t {
  float3 orig, dir;
  float t;
} ray_t;

typedef struct tag_pathtrace_params {
  float max_extinction;
} pathtrace_params;

typedef struct THClState {
  int initialized;
  int allowNonGpus;
  int allocatedDevices;
  int currentDevice;
  int trace;

  int addFinish;

  int detailedTimings;
  struct THClScratchSpace **scratchSpaceByDevice;

  struct DeviceInfo **deviceInfoByDevice;

  struct EasyCL **clByDevice;

} THClState;

typedef struct {
  unsigned int tex1Index, tex2Index;
} ScaleTexParam;

typedef struct {
  int n, lda, j0, npivots;
  short ipiv[32];
} dlaswp_params_t2;

typedef struct {
  float2 amod;
  float2 bs;
  float2 bis;
} processed_line2;

typedef struct {
  float fx;
  float fy;
  float fz;
  int uw;
} b3AABBCL;

typedef struct {
  float3 vertex;
  float3 normal;
  float2 texCoord;
} VertexData;

typedef struct pdf14_tmp {
  unsigned int digest[4];
  unsigned int out[4];

} pdf14_tmp_t;

typedef struct _qshiftData {
  double qH2Q, qH2G, drH2Q;
} qshiftData_t;

typedef struct _intinfo intinfo;

struct _intinfo {
  unsigned int num_integrals;

  global const unsigned int *rows;

  global const unsigned int *cols;

  global const unsigned int *cidx;
};

typedef struct lotus8_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[5];
  unsigned int out[5];

} lotus8_tmp_t;

typedef struct {
  int vectors_cnt;
  int countX;
  int countY;
  int countZ;
  int obj_num;
} cl_params_t;

typedef struct bestcrypt_tmp {
  unsigned int salt_pw_buf[33];
  unsigned int out[8];

} bestcrypt_tmp_t;

typedef struct {
  int constant_field;
} CommonTypedef;

struct coprthr_td_attr {
  int detachstate;
  int initstate;
  int dd;
};

typedef struct {
  float x;
  float y;
  float z;
  float w;
} ApocVector;

typedef struct {
  unsigned int s[16];
  unsigned int i;
} well512_state;

typedef struct {
  char data[6];
} w_vec_data;

typedef struct jwt {
  unsigned int salt_buf[1024];
  unsigned int salt_len;

  unsigned int signature_len;

} jwt_t;

typedef struct {
  float3 min;
  float3 max;

  float invTransform[16];
  union {
    struct {
      unsigned int leftChildIndex;
      unsigned int rightChildIndex;
    };
    unsigned int subBvh;
  };
  unsigned int isLeaf;
} TopBvhNode;

typedef struct ecryptfs_tmp {
  unsigned long long out[8];

} ecryptfs_tmp_t;

typedef struct {
  uint32_t base;
  uint32_t stride;
  uint32_t head;
  uint32_t tail;
} clIndexedQueue_32;

typedef struct {
  const char *vendor_name;
  cl_context existing_context;
  const char *compile_options;
  cl_command_queue_properties default_queue_props;
  cl_context_properties *default_context_props;
  cl_device_type preferred_device_type;
} clu_initialize_params;

typedef struct {

  float3 diffuse;

  float kd;

  float3 extinction;

  float kt;

  float3 emission;

  float emission_power;

  float ks;

  float specExp;

  float ior;

  float refExp;
} material_t;

typedef struct BVHTreeNode {

  CL_AABB aabb;
  int2 packet_indexes;
  int2 padding;
} BVHTreeNode;

typedef struct s {
  void *f;
} s_t;

typedef struct {

  bbox bounds;

  float3 voxelsize;

  float3 voxelsizeinv;

  int4 gridres;
} GridDesc;

typedef struct {
  float bias;
  float alpha;
  float beta;
} BiasBnParam;

typedef struct {
  SUM_SSE sum_sse[64][64];
} GPU_SCRATCH;

typedef struct {
  union {
    struct {
      float3 point;
      float3 normal;
    };
    struct {
      float pnt[3];
      float depth;
      float nrm[3];
      unsigned int matIndex;
    };
  };
} RayIntersect;

typedef struct {
  int type;
  int rows;
  int cols;
  union {
    uchar *ptr;
    short *s;
    int *i;
    float *fl;
    double *db;
  } host;
  cl_mem device_mem;
  size_t stride;
  size_t matrix_memory_size;
} CvGpuMat;

typedef struct {
  int _wcl_field;
} _wcl_typedef;

typedef struct _PAPhysicsNormalData {
  float strength, noise;
  PAPhysicsFalloffData falloff;
} PAPhysicsNormalData;

typedef struct {
  int cache_write_addr;
  float write_data[(
      (((((((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
             (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) |
           (((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
              (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) >>
            (4)))) |
         (((((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
              (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) |
            (((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
               (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) >>
             (4)))) >>
          (8)))) |
       (((((((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
              (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) |
            (((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
               (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) >>
             (4)))) |
          (((((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
               (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) |
             (((((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) |
                (((((3 + 5 - 1) - 1) | (((3 + 5 - 1) - 1) >> (1)))) >> (2)))) >>
              (4)))) >>
           (8)))) >>
        (16))) +
      1)][(((((((((((16 - 1) | ((16 - 1) >> (1)))) |
                  ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) |
                ((((((16 - 1) | ((16 - 1) >> (1)))) |
                   ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) >>
                 (4)))) |
              ((((((((16 - 1) | ((16 - 1) >> (1)))) |
                   ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) |
                 ((((((16 - 1) | ((16 - 1) >> (1)))) |
                    ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) >>
                  (4)))) >>
               (8)))) |
            ((((((((((16 - 1) | ((16 - 1) >> (1)))) |
                   ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) |
                 ((((((16 - 1) | ((16 - 1) >> (1)))) |
                    ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) >>
                  (4)))) |
               ((((((((16 - 1) | ((16 - 1) >> (1)))) |
                    ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) |
                  ((((((16 - 1) | ((16 - 1) >> (1)))) |
                     ((((16 - 1) | ((16 - 1) >> (1)))) >> (2)))) >>
                   (4)))) >>
                (8)))) >>
             (16))) +
           1)];
} PoolTailOutput;

typedef struct {
  float3 pMin;
  float3 pMax;
  float tMin;
  float tMax;
} AABB;

typedef struct BVHNodeGPU {
  AABB aabb;
  int vert_list[10];
  int child_idx;
  int vert_len;
} BVHNodeGPU;

typedef struct {
  unsigned int tex1Index, tex2Index;
} DotProductTexParam;

typedef struct {
  PerspectiveCamera perspCamera;

  Transform leftEyeRasterToCamera;
  Transform leftEyeCameraToWorld;

  Transform rightEyeRasterToCamera;
  Transform rightEyeCameraToWorld;
} StereoCamera;

typedef struct _curveData {
  char *id;
  char *filename;
  int weight;
  double normalized_weight;
  double alpha1, beta1, gamma1, alpha2, beta2, gamma2;
  int nPoints;
  double *r, *input, *output;
} curveData_t;

typedef struct ZeroFuncs {
  char names[64][64];
} ZeroFuncs;

typedef struct {
  uint8_t mblim[16];
  uint8_t lim[16];
  uint8_t hev_thr[16];
} loop_filter_thresh_ocl;

typedef struct {
  double *buf;
  int dim[2];
} __PSGrid2DDoubleDev;

typedef struct {
  union {
    float data[4];
    float rgba[4];
    struct {
      float x;
      float y;
      float z;
      float w;
    };
    struct {
      float r;
      float g;
      float b;
      float a;
    };
  };
} PointXYZRGBA;

typedef struct {
  float3 lo, hi;
} v2float3;

typedef struct {
  int (*int_array_pointer)[1];
  char (*char_array_pointer)[1];
  float (*float_array_pointer)[1];
  float4 (*vector_array_pointer)[1];
} PrivateArrayPointerStruct;

typedef struct lidar {
  float4 coeff;
  float min_distance;

} lidar;

typedef struct {
  float3 v0;
  float3 v1;
  float3 v2;
} triangle_verts;

typedef struct VoxelBuffer {

  float3 minPos;

  float3 maxPos;

  Voxel grid[256 * 256 * 256];
} VoxelBuffer;

typedef struct {
  ddr_weight_wng lane[32];
} channel_vec_wng;

typedef struct KernelParams {
  float4 ps, c0;
  float4 maxidx;
  uint4 dimlen, cp0, cp1;
  uint2 cachebox;
  float minstep;
  float twin0, twin1, tmax;
  float oneoverc0;
  unsigned int isrowmajor, save2pt, doreflect, dorefint, savedet;
  float Rtstep;
  float minenergy;
  float skipradius2;
  float minaccumtime;
  unsigned int maxdetphoton;
  unsigned int maxmedia;
  unsigned int detnum;
  unsigned int idx1dorig;
  unsigned int mediaidorig;
  unsigned int blockphoton;
  unsigned int blockextra;
} MCXParam __attribute__((aligned(32)));

typedef struct {
  int nOffset;
  int kOffset;
} BinomialCoefficientContext;

typedef struct {
  enum DDBFlags {
    DDB_HAS_DATA_INFO = 1,
    DDB_SCHEDULER_PROFILING = 2,
    DDB_COMMAND_QUEUE_RAW = 4
  } ddbFlags;
  unsigned int m_size;
  unsigned int m_stackTop;
  unsigned int m_dataInfoTop;
  unsigned int m_stackBottom;
  unsigned int m_dataInfoBottom;
  unsigned int m_dataInfoSize;
  unsigned int m_flags;

  unsigned int m_offset;
  unsigned int m_data[100];
} DebugDataBuffer;

typedef struct __attribute__((aligned(16))) {
  float3 min;
  float3 max;
  unsigned int objIdx;
} LeafNode;

typedef struct _mat4 {
  float4 x, y, z, w;
} mat4;

typedef union {
  int count;
  char *filename;
} intechar;

typedef struct _fileNode {
  intechar data;

  struct _fileNode *next;
} fileNode_t;

typedef struct {
  unsigned int strategy_first;
  unsigned int strategy_second;
  unsigned int past_behavior;
  unsigned int score;
} player;

typedef struct {
  cl_mem *data;
} scalar_buffer;

typedef struct bc7_enc_settings {
  bool mode_selection[4];
  int refineIterations[8];

  bool skip_mode2;
  int fastSkipTreshold_mode1;
  int fastSkipTreshold_mode3;
  int fastSkipTreshold_mode7;

  int mode45_channel0;
  int refineIterations_channel;

  int channels;
  unsigned int width, height, stride;
} bc7_enc_settings;

typedef struct {
  float4 m0;
  float4 m1;
  float4 m2;
  float4 m3;
} matrix4x4;

typedef struct __attribute__((aligned(16))) {
  float output;
  float bias;
  float error;

  int numOfInputs;
  int reccurent;
} Perceptron;

typedef struct {
  float4 a;
  float4 b;
} line4;

typedef struct float11 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
} float11;

typedef struct float2 {
  double x;
  double y;
} complex;

typedef struct s_bundle {
  int idcount;
  int *ids;
  int *matches_counts;
  int **matches;
  int *local_counts;
  t_body **locals;
  int *cell_sizes;
  cl_float4 **cells;
  int cellcount;
  int index;
} t_bundle;

typedef struct {
  char c;
  void *v;
  void *v2;
} my_st;

typedef struct {
  char profile[256];
  char version[256];
  char name[256];
  char vendor[256];
  char extensions[(256 * 128)];
} clu_platform_info;

typedef struct {
  float absoluteTheta;
  float absolutePhi;
  float zenith_Y, zenith_x, zenith_y;
  float perez_Y[6], perez_x[6], perez_y[6];
} SkyLightParam;

typedef struct {
  unsigned int buffer[16];
} md5_ctx;

typedef struct {
  uint64_t d[8];
  uint32_t size;
} mpzcls_t;

typedef struct TfLiteDelegateParams {
  struct TfLiteDelegate *delegate;
  TfLiteIntArray *nodes_to_replace;
  TfLiteIntArray *input_tensors;
  TfLiteIntArray *output_tensors;
} TfLiteDelegateParams;

typedef struct MP_GridCG {

  double *dval;
  double *g;
  double *h;
} MP_GridCG;

typedef struct def_FluidInfo {

  float mass;

  float k_gas;
  float k_viscosity;
  float rest_density;
  float sigma;
  float k_threshold;
  float k_wall_damper;
  float k_wall_friction;

  float3 gravity;
} FluidInfo;

typedef struct {
  float2 e0;
  float2 e1;
  float2 e2;
} su3vec __attribute__((aligned(16)));

typedef struct __attribute__((aligned(4))) {
  ProceduralTextureHead head;
} Float3Random1Texture;

typedef struct struct_type {
  float4 float4d;
  int intd;
} typedef_struct_type;

typedef struct {
  pbkdf2_salt pbkdf2;
  uchar data[1024];
} strip_salt;

typedef struct {
  Vector absoluteSunDir, absoluteUpDir;
  Spectrum aTerm, bTerm, cTerm, dTerm, eTerm, fTerm, gTerm, hTerm, iTerm,
      radianceTerm;
  int hasGround, isGroundBlack;
  Spectrum scaledGroundColor;
  unsigned int distributionOffset;
  int useVisibilityMapCache;
} SkyLight2Param;

typedef struct {
  Vector absolutePos;
  Spectrum emittedFactor;
} PointLightParam;

typedef struct {
  Vector absolutePos, localPos;
  Spectrum emittedFactor;
  float avarage;
  unsigned int imageMapIndex;
} MapPointLightParam;

typedef struct {
  Vector absolutePos;
  Spectrum emittedFactor;
  float cosTotalWidth, cosFalloffStart;
} SpotLightParam;

typedef struct {
  Vector absolutePos, lightNormal;
  Spectrum emittedFactor;
  Matrix4x4 lightProjection;
  float screenX0, screenX1, screenY0, screenY1;
  unsigned int imageMapIndex;
} ProjectionLightParam;

typedef struct {
  Spectrum float3;
  Vector absoluteLightDir, x, y;
  float cosThetaMax;
} DistantLightParam;

typedef struct {
  Vector absolutePos;
  Spectrum emittedFactor;
  float radius;
} SphereLightParam;

typedef struct {
  unsigned int kdTexIndex;
  unsigned int ksTexIndex;
  unsigned int nuTexIndex;
  unsigned int nvTexIndex;
  unsigned int kaTexIndex;
  unsigned int depthTexIndex;
  unsigned int indexTexIndex;
  int multibounce;
} Glossy2Param;

typedef struct {
  unsigned int start_index;
  unsigned int count;
} CellData;

typedef struct {
  SphereLightParam sphere;
  float avarage;
  unsigned int imageMapIndex;
} MapSphereLightParam;

typedef struct {
  Transform light2World;
  Spectrum gain, temperatureScale;

  union {
    SunLightParam sun;
    SkyLight2Param sky2;
    InfiniteLightParam infinite;
    PointLightParam point;
    MapPointLightParam mapPoint;
    SpotLightParam spot;
    ProjectionLightParam projection;
    ConstantInfiniteLightParam constantInfinite;
    SharpDistantLightParam sharpDistant;
    DistantLightParam distant;
    LaserLightParam laser;
    SphereLightParam sphere;
    MapSphereLightParam mapSphere;
  };
} NotIntersectableLightSource;

typedef struct {
  w_vec_data lane[32];
} channel_wvec_wng;

typedef struct {
  Vector p;
  Normal n;
  unsigned int outgoingRadianceIndex;
  int isVolume;
} RadiancePhoton;

typedef struct {
  int vector_x;
  int vector_y;
} vector_net;

typedef struct md4_hmac_ctx {
  md4_ctx_t ipad;
  md4_ctx_t opad;

} md4_hmac_ctx_t;

typedef struct {
  unsigned int x;
  unsigned int c;
} mwc64x_state_t;
typedef struct {
  int numOfLayers;
  __local int *layers;
  bool classification;
  float minOutputValue;
  float maxOutputValue;
  float learningFactor;
  float lambda;
} setup_t;

typedef struct {
  int y_ac_i;
  int y_dc_idelta;
  int y2_dc_idelta;
  int y2_ac_idelta;
  int uv_dc_idelta;
  int uv_ac_idelta;
  int loop_filter_level;
  int mbedge_limit;
  int sub_bedge_limit;
  int interior_limit;
  int hev_threshold;
} segment_data;

typedef struct {
  lane_data wvec[8];
} data_wng_vec8;

typedef struct {
  char v0;
  int v1;
  float v2;
} boost_tuple_char_int_float_t;

typedef struct {
  int orig_id;
  float3 strength;
  float3 start_p, in_dir;
  int4 geometry;
  float intersect_number;
  float3 intersect_p, normal;
  float optical_density;
  float shininess;
} unit_data;

typedef struct {
  uint32_t iterations;
  uint32_t outlen;
  uint32_t skip_bytes;
  uint8_t aes_ct[256];
  uint32_t aes_len;
  uint8_t iv[16];
  uint8_t salt[64];
  uint8_t length;
} odf_salt;

typedef struct _derivatives_t {
  float3 do_dx, do_dy, dd_dx, dd_dy;
  float2 duv_dx, duv_dy;
  float3 dndx, dndy;
  float ddn_dx, ddn_dy;
} derivatives_t;

typedef struct b3BvhSubtreeInfoData b3BvhSubtreeInfoData_t;

struct b3BvhSubtreeInfoData {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_rootNodeIndex;

  int m_subtreeSize;
  int m_padding[3];
};

typedef struct {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_escapeIndexOrTriangleIndex;
} b3QuantizedBvhNode;

typedef struct {
  unsigned int s1, s2, s3;
} taus_state_t;

typedef struct taus_uint4_ {
  unsigned int s0;
  taus_state_t state;
} taus_uint4;

typedef struct tag_float66 {
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
} float66;

typedef struct {
  int partition_count;
  float4 endpt0[4];
  float4 endpt1[4];
} endpoints;

typedef struct {
  endpoints ep;
  float weights[216];
  float weight_error_scale[216];
} endpoints_and_weights;

typedef struct b2clSweep {
  float2 localCenter;
  float2 c0, c;
  float a0, a;

  float alpha0;
  float dummy;
} b2clSweep;

typedef struct {
  uint64_t head;
  uint64_t free;
} clqueue_64;

typedef struct b2clBodyDynamic {
  b2clSweep m_sweep;

  float2 m_linearVelocity;
  float2 m_force;
  float m_angularVelocity;
  float m_torque;
  int m_last_uid;
  int dummy;
} b2clBodyDynamic;

typedef struct Microfacet {
  float F;
  float G;
  float D;
} Microfacet;

typedef struct {
  uint32_t mults[(128 * 4 * 1024 / 16)];
  uint32_t hash[(128 * 4 * 1024 / 16)];
  int type[(128 * 4 * 1024 / 16)];
  uint32_t prime[(128 * 4 * 1024 / 16)];
} result_t;

typedef struct {
  Spectrum rgb;
} IrregularDataParam;

typedef struct {
  float m_min[2];
  float m_max[2];
  uchar m_sType;
  uchar m_bType;

} b2clAABB;

typedef struct DecompMotionTransform {
  Transform mid;
  float4 pre_x, pre_y;
  float4 post_x, post_y;
} DecompMotionTransform;

typedef struct whirlpool_hmac_ctx_vector {
  whirlpool_ctx_vector_t ipad;
  whirlpool_ctx_vector_t opad;

} whirlpool_hmac_ctx_vector_t;

typedef struct HemisphereSampler {
  float4 basis_x;
  float4 basis_y;
  float4 basis_z;
  uint4 direction;
} HemisphereSampler;

typedef struct androidfde {
  unsigned int data[384];

} androidfde_t;

typedef struct {
  unsigned int num_uncracked_hashes;
  unsigned int offset_table_size;
  unsigned int hash_table_size;
  unsigned int bitmap_size_bits;
  unsigned int cmp_steps;
  unsigned int cmp_bits;
} DES_hash_check_params;

typedef struct intintIdentitySentinelPerfectCLHash_Bucket {
  int value;
} intintIdentitySentinelPerfectCLHash_Bucket;

typedef struct {
  cl_float3 start;
  cl_float3 dir;
  Vector d;
} Ray;

typedef struct IntersectionInfo {
  bool m_has_intersected;
  float3 m_pos;
  float3 m_normal;
  float m_angle_between;
  Ray m_incoming_ray;
  Material m_material;
} IntersectionInfo;

typedef struct {
  float v[16];
} ReluChannelVector;

typedef struct {
  float4 m_pos;
  float4 m_quat;
  float4 m_linVel;
  float4 m_angVel;

  unsigned int m_shapeIdx;
  float m_invMass;
  float m_restituitionCoeff;
  float m_frictionCoeff;
} b3RigidBodyCL;

typedef struct ListNode {
  struct ListNode *prev;
  struct ListNode *next;
} ListNode;

typedef struct atmi_data_s {

  void *ptr;

  unsigned int size;

  atmi_mem_place_t place;

} atmi_data_t;

typedef struct sha512_hmac_ctx_vector {
  sha512_ctx_vector_t ipad;
  sha512_ctx_vector_t opad;

} sha512_hmac_ctx_vector_t;

typedef struct {
  unsigned int dummy;
} WclConstants;

typedef struct {
  __constant WclConstants *wcl_constant_allocations_min;
  __constant WclConstants *wcl_constant_allocations_max;
  __constant unsigned int *stream_count_kernel__unsorted_elements_min;
  __constant unsigned int *stream_count_kernel__unsorted_elements_max;
} WclConstantLimits;

typedef struct {
  float3 row0;
  float3 row1;
  float3 row2;
} mat3;

typedef struct {
  unsigned int rgbOffset, alphaOffset;
  unsigned int width, height;
} TexMap;

typedef struct __attribute__((aligned(8))) {
  unsigned int magic;
  unsigned int fmtoffset;
  int globalId;
} clPrintfHeader;

typedef struct _control {
  float angles[5];
} control;

typedef struct {
  uint32_t cracked;
} keychain_out;

typedef struct {
  int taskData_b_offset_learningInputs, taskData_b_offset_learningOutputs;
  int taskData_totalLearningLines;
  int taskData_b_offset_testInputs, taskData_b_offset_testOutputs;
  int taskData_totalTestLines;
  int taskData_b_size;
} task_data_transform_t;

typedef struct LightParameters {
  float3 position;
  float3 ambientColor;
  float3 diffuseColor;
  float3 specularColor;
  float specularExponent;
  int shadingMode;

  char padding[56];
} LightParameters;

typedef struct {
  float3 vertices[3];
  Material material;
} EmissiveTriangle;

typedef struct {
  Ray ray;
  Vector vVector, hVector;
} Viewport;

typedef struct {

  volatile unsigned int next;
} clqueue_item;

typedef struct {
  type_simbolo t;
  int proximo;
} t_item_programa;

typedef struct {
  Spectrum float3;
} ConstFloat3Param;

typedef struct {
  int lower_bound;
  int upper_bound;
  int loop_step;
  int chunk_size;
} nanos_ws_info_loop_t;

typedef struct {

  int startidx;

  int startvtx;

  int volume_idx;

  int id;

  float3 linearvelocity;

  float4 angularvelocity;

  matrix4x4 transform;
  Material material;
} Shape;

typedef struct {
  unsigned int aa_lvl;
  unsigned int aa_dim;
  float aa_inc;
  float aa_div;
  unsigned int pixels_X;
  unsigned int t_depth;
  float cam_foc;
  float cam_apt;
  float3 cam_pos;
  float3 cam_ori;
  float3 cam_fwd;
  float3 cam_rgt;
  float3 cam_up;
  float3 bl_ray;
} RenderInfo;

typedef struct {
  bool is_final;
  bool reserved1;
  bool reserved2;
  bool reserved3;
  bool reserved4;
  bool reserved5;
  bool reserved6;
  bool reserved7;
} nanos_wd_dyn_flags_t;

typedef struct {
  float2 c;
  float2 n;
  float2 s;
  float2 e;
  float2 w;
  float2 u;
  float2 d;
} Neighbors_float2;

typedef struct {
  unsigned int key;
} keystore_hash;

typedef struct bucket bucket;
struct bucket {
  struct bucket *link;
  struct bucket *next;
  char *name;
  char *tag;
  short value;
  short index;
  short prec;
  char class;
  char assoc;
};

typedef struct {
  float r;
  float i;
} comp;

typedef struct LSX_t {
  int64_t mx00, mx01, mx11, my0, my1;
  short2 dirSE, vDirSE;
  int distSquSE, padding;
} LSX_t;

typedef struct md5_ctx_vector {
  unsigned int h[4];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} md5_ctx_vector_t;

typedef struct md5_hmac_ctx_vector {
  md5_ctx_vector_t ipad;
  md5_ctx_vector_t opad;

} md5_hmac_ctx_vector_t;

typedef struct sha256_hmac_ctx {
  sha256_ctx_t ipad;
  sha256_ctx_t opad;

} sha256_hmac_ctx_t;

typedef struct _FileCallBackRec {
  int thefileid;
  int theattid;
  int thevar;
} FileCallBackRec;

typedef struct {
  float3 point;
  float lastDist;
  float depth;
  float distThresh;
  int objectId;
  bool found;
  int count;
} sRayMarchingOut;

typedef struct {
  float u, v;
} UV;

typedef struct {
  Vector X, Y, Z;
} Frame;

typedef struct {

  Vector fixedDir;
  float4 hitPoint;
  UV hitPointUV;
  Normal geometryN;
  Normal shadeN;

  unsigned int materialIndex;

  Frame frame;

} BSDF;

typedef struct {
  float passThroughEvent;
  BSDF passThroughBsdf;
} PathStateDirectLightPassThrough;

typedef struct pem {
  unsigned int data_buf[16384];
  int data_len;

  int cipher;

} pem_t;

typedef struct {
  unsigned int depth, diffuseDepth, glossyDepth, specularDepth;
} PathDepthInfo;

typedef struct {
  PathDepthInfo depth;
  PathVolumeInfo volume;

  int isPassThroughPath;

  int lastBSDFEvent;
  float lastBSDFPdfW;
  float lastGlossiness;
  Normal lastShadeN;
  bool lastFromVolume;

  int isNearlyCaustic;

  int isNearlyS, isNearlySD, isNearlySDS;
} EyePathInfo;

typedef struct {
  int3 s0, s1, s2, s3;
} v4int3;

typedef struct _pbc {
  double basis[3][3];
  double reciprocal_basis[3][3];
  double cutoff;
  double volume;
} pbc_t;

typedef struct workItemResult_t {
  int2 mv_predicted;
  int2 mv_diff;
  unsigned int costs;
  char refIdx;
} workItemResult __attribute__((aligned));

typedef struct {

  float4 p;
  Normal n;
  int isVolume;

  unsigned int distributionOffset;
} ELVCacheEntry;

typedef struct {
  int first;
  float second;
} _pair_int_float_t;

typedef struct {
  RSComplex MP_EA;
  RSComplex MP_RT;
  RSComplex MP_RA;
  RSComplex MP_RF;
  short int l_value;
} Pole;

typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
  unsigned char alpha;
} RGB32;

typedef struct {
  RGB32 colors[4];
} RGB32_x4;

typedef struct {
  double8 lo, hi;
} v2double8;

typedef struct inner {
  float4 x;
} inner;

typedef struct {
  unsigned int ostack[1024];
  unsigned char tstack[1024];
} CastStackz;

typedef struct office2010 {
  unsigned int encryptedVerifier[4];
  unsigned int encryptedVerifierHash[8];

} office2010_t;

typedef struct {
  float v[16];
} DotVector;

typedef struct {
  DotVector data[(3 + 5 - 1)];
} InputReaderOutput;

typedef struct android_backup_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} android_backup_tmp_t;

typedef struct {
  int pos;
  unsigned char data;
} Channel_type;

typedef struct {

  int numRegistros;
  int numVariaveis;
  float *registros;

} Database;

typedef struct {
  long long ll[11];
} XXH64_state_t;

typedef struct b3Collidable b3Collidable_t;

struct b3Collidable {
  union {
    int m_numChildShapes;
    int m_bvhIndex;
  };
  union {
    float m_radius;
    int m_compoundBvhIndex;
  };

  int m_shapeType;
  union {
    int m_shapeIndex;
    float m_height;
  };
};

typedef struct {
  union {
    struct {

      float bboxMin[3];
      float bboxMax[3];
    } bvhNode;
    struct {
      unsigned int v[3];
      unsigned int meshIndex, triangleIndex;
    } triangleLeaf;
    struct {
      unsigned int leafIndex;
      unsigned int transformIndex, motionIndex;
      unsigned int meshOffsetIndex;
    } bvhLeaf;
  };

  unsigned int nodeData;
  int pad0;
} BVHAccelArrayNode;

typedef struct PointPair {

  float2 p1;

  float2 p2;
} PointPair;

typedef struct MotionVector {
  PointPair p;

  int should_consider;
} MotionVector;

typedef struct struct_of_arrays_arg {
  int i1[2];
  float f1;
  int i2[4];
  float f2[3];
  int i3;
} struct_of_arrays_arg_t;

typedef struct VoxelPoint {

  unsigned int x;
  unsigned int y;
  unsigned int z;
  uchar plane;
} VoxelPoint;

typedef struct __Distribution2D {
  int w;
  int h;
  __global float const *data;
} Distribution2D;

typedef struct {
  int genes[256];
} __SimpleChromosome;

typedef struct {
  float t;
  float u;
  float v;
} triangle_inter;

typedef struct {
  triangle_inter inter;
  unsigned int index;
} intersection;

typedef struct vc64_sbog_tmp {
  unsigned long long ipad_raw[8];
  unsigned long long opad_raw[8];

  unsigned long long ipad_hash[8];
  unsigned long long opad_hash[8];

  unsigned long long dgst[32];
  unsigned long long out[32];

  unsigned long long pim_key[32];
  int pim;

} vc64_sbog_tmp_t;

typedef struct pwsafe3_tmp {
  unsigned int digest_buf[8];

} pwsafe3_tmp_t;

typedef struct md4_ctx_vector {
  unsigned int h[4];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} md4_ctx_vector_t;

typedef struct _test_arg_struct {
  int a;
  int b;
} test_arg_struct;

typedef struct {
  void *original;
  void *privates;
  size_t element_size;
  size_t num_scalars;
  void *descriptor;
  void *bop;
  void *vop;
  void *cleanup;
} nanos_reduction_t;

typedef struct type_parameters {
  int generation;
  int population;
  int width;
  int height;
  int elite;
  int numberOfRectangles;
} Parameters;

typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;

typedef struct {
  unsigned int particles_count;
  float max_velocity;
  float fluid_density;
  float total_mass;
  float particle_mass;
  float dynamic_viscosity;
  float simulation_time;
  float target_fps;
  float h;
  float simulation_scale;
  float time_delta;
  float surface_tension_threshold;
  float surface_tension;
  float restitution;
  float K;

  float3 constant_acceleration;

  int grid_size_x;
  int grid_size_y;
  int grid_size_z;
  unsigned int grid_cell_count;
  float3 min_point, max_point;
} simulation_parameters;

typedef struct {
  int offs[4];
  int strds[4];
  char isSeq[4];
} AssignKernelParam_t;

typedef struct {
  unsigned int AIndex;
  unsigned int BIndex;
  float Distance;
} SortItem;

typedef struct {
  int width;
  int height;
  float2 sp;

} RT_ViewPlane;

typedef struct {
  int size;
  int uniqueElements;
  int subsetSize;
} CombinatoricContext;

typedef struct {
  bbox lbound;
  bbox rbound;
} FatBvhNode;

typedef struct sha1_hmac_ctx_vector {
  sha1_ctx_vector_t ipad;
  sha1_ctx_vector_t opad;

} sha1_hmac_ctx_vector_t;

typedef struct _DifferentialGeometry {

  float3 p;

  float3 n;

  float3 ng;

  float2 uv;

  float3 dpdu;
  float3 dpdv;

  matrix4x4 world_to_tangent;
  matrix4x4 tangent_to_world;

  Material mat;
  float area;
  int transfer_mode;
  int padding[2];
} DifferentialGeometry;

typedef struct Inspector {
  float isoThreshold;
  int numberOfValidCubes;
  int numberOfValidPoints;
  int dummy[1];
} Inspector;

typedef struct {
  void *buf;
  int dim[1];
} __PSGrid1DDev;

typedef struct image2d_t_ {
  uchar4 *data;
  int width;
  int height;
  int rowpitch;
  int order;
  int data_type;
} image2d_t_;

typedef struct float7 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
} float7;

typedef struct _ptemp {
  int *index;
  double *templist;
} ptemp_t;

typedef struct {
  float3 gravity;
  float timeStep;
  float relaxationFactor;
  float gridSize;
  float mass;
  float radius;
  float restDensity;
  float interactionRadius;

  float monaghanSplineNormalisation;
  float monaghanSplinePrimeNormalisation;

  float surfaceTension;
  float surfaceTensionTerm;
  float surfaceTensionNormalisation;

  float adhesion;
  float adhesionNormalisation;

  float viscosity;
} fluid_t;

struct DecodeVariables {

  int p_jinfo_smp_fact;

  unsigned int p_jinfo_quant_tbl_quantval[4][64];

  int p_jinfo_dc_xhuff_tbl_bits[2][36];
  int p_jinfo_dc_xhuff_tbl_huffval[2][257];

  int p_jinfo_ac_xhuff_tbl_bits[2][36];
  int p_jinfo_ac_xhuff_tbl_huffval[2][257];

  int p_jinfo_dc_dhuff_tbl_ml[2];
  int p_jinfo_dc_dhuff_tbl_maxcode[2][36];
  int p_jinfo_dc_dhuff_tbl_mincode[2][36];
  int p_jinfo_dc_dhuff_tbl_valptr[2][36];

  int p_jinfo_ac_dhuff_tbl_ml[2];
  int p_jinfo_ac_dhuff_tbl_maxcode[2][36];
  int p_jinfo_ac_dhuff_tbl_mincode[2][36];
  int p_jinfo_ac_dhuff_tbl_valptr[2][36];

  int p_jinfo_MCUWidth;
  int p_jinfo_MCUHeight;
  int p_jinfo_NumMCU;

  int rgb_buf[4][3][64];
  short p_jinfo_image_height;
  short p_jinfo_image_width;
  char p_jinfo_data_precision;

  char p_jinfo_num_components;

  unsigned int p_jinfo_jpeg_data;
  char p_jinfo_comps_info_index[3];
  char p_jinfo_comps_info_id[3];
  char p_jinfo_comps_info_h_samp_factor[3];
  char p_jinfo_comps_info_v_samp_factor[3];
  char p_jinfo_comps_info_quant_tbl_no[3];
  char p_jinfo_comps_info_dc_tbl_no[3];
  char p_jinfo_comps_info_ac_tbl_no[3];
};

typedef struct _neural_vector_tag {
  unsigned int feature_offset;
  unsigned int spatial_offset;
  unsigned int raw_size;
  unsigned int data[1];
} neural_vector;

typedef struct {
  float scaleX;
  float scaleY;
  float dx;
  float dy;
  float angleX;
  float angleY;
} ellipseData;

typedef struct {
  uint32_t total[2 * 1024];
  uint32_t state[8 * 1024];
  uint32_t buffer[16 * 1024];
  uint32_t W[64 * 1024];
  uint32_t msglen[2 * 1024];
  uint32_t padding[16 * 1024];
} sha256cl_context;

typedef struct {

  float8 point;
  float8 field;

} node_field_t;

typedef struct secp256k1 {
  unsigned int xy[96];

} secp256k1_t;

typedef struct {
  salt_t pbkdf2;
  uint32_t hmacdatalen;
  uint8_t hmacdata[2048];
  uint8_t expectedhmac[16];
} cloudkeychain_salt_t;

typedef struct {
  unsigned int len;
  ushort password[256];
} mid_t;

typedef struct {
  float mass;
} RigidBodyProperties;

typedef struct {
  float3 o;
  float r;
} Sph;

typedef struct {
  w_vec_pool lane[32];
} pool_vec_wng;

typedef struct s_ckparam {

  int sampling_frequency;

  int N;

  int tracksize;

  int start_sample;
  int stop_sample;
} t_ckparam;

typedef struct {
  int shape_id;
  int prim_id;
  int2 padding;

  float4 uvwt;
} Intersection;

typedef struct {
  float2 min;
  float2 max;
} Rect;

typedef struct Tensor3D {
  __global uchar *ptr;
  int offset_first_element_in_bytes;
  int stride_x;
  int stride_y;
  int stride_z;
} Tensor3D;

typedef struct {
  unsigned int W[5];
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int out[5];
  unsigned int partial[5];
} wpapsk_state;

struct gzFile_s {
  unsigned have;
  unsigned char *next;
  long pos;
};

typedef struct {

  float3 throughput;

  unsigned int subpixelIndex;
  unsigned int pixelIndex;
  unsigned int depth;
  unsigned int flags;
  unsigned int seed;

  unsigned int _reserved1;
  unsigned int _reserved2;
} Path;

typedef struct TF_TString_Small {
  uint8_t size;
  char str[64 + sizeof(char)];
} TF_TString_Small;

typedef struct TF_TString_Offset {
  uint32_t size;
  uint32_t offset;
  uint32_t count;
} TF_TString_Offset;

typedef struct TF_TString_View {
  size_t size;
  const char *ptr;
} TF_TString_View;

typedef struct TF_TString_Raw {
  uint8_t raw[24];
} TF_TString_Raw;

typedef struct TF_TString {
  union {

    TF_TString_Small smll;
    TF_TString_Large large;
    TF_TString_Offset offset;
    TF_TString_View view;
    TF_TString_Raw raw;
  } u;
} TF_TString;

typedef struct _NodeHashEntry {
  ushort poslo;
  ushort poshi;
  ushort size;
} NodeHashEntry;

typedef struct {
  unsigned int array[3];
} boundary_index_array_3;

typedef struct BlkmulArgNames {
  const char *coordA;
  const char *coordB;
  const char *skewRow;
  const char *skewCol;
  const char *k;
  const char *vectBoundK;
} BlkmulArgNames;

typedef struct {
  uint64_t objSize;
  clqueue_64 queue;
  uint64_t heap;
  uint64_t reduce_mem;
} clArrayList_64;

typedef struct half9 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
} half9;

typedef struct {
  float z_min;
  float z_max;

  float mutr;
  float mua;
  float g;
  float n;

  float flc;
  float muaf;
  float eY;
  float albedof;
} LayerStruct;

typedef struct {
  DPTYPE_PE_VEC lane[8];
} SCAL_PE_VEC;

typedef struct sha1_ctx {
  unsigned int h[5];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} sha1_ctx_t;

typedef struct arg_da_struct_type {
  float *r;
  float *g;
  float *b;
  float *x;
  float *d_r;
  float *d_g;
  float *d_b;
  float *d_x;
  int start_index;
  int end_index;
} args_da;

typedef struct {
  DotVector v[3];
} DotFilterVector;

typedef struct {
  DotFilterVector filter_data;
  BiasBnParam bias_bn_data;
  int cache_addr;
  int n_inc;
} FilterReaderOutput;

typedef struct {
  unsigned int surface;
  unsigned int v0;
  unsigned int v1;
  unsigned int v2;
} triangle;

typedef struct half0 {
  float s0;
} half0;

typedef struct {
  float startTime, endTime;
  Transform start, end;
  DecomposedTransform startT, endT;
  float4 startQ, endQ;
  int hasRotation, hasTranslation, hasScale;
  int hasTranslationX, hasTranslationY, hasTranslationZ;
  int hasScaleX, hasScaleY, hasScaleZ;

  int isActive;
} InterpolatedTransform;

typedef struct half11 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
} half11;

typedef struct {
  float blood;
  float ist;
  float time;
} mvalue;

typedef struct {
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
} cl_handle_t;

typedef struct {
  int p;
  int n;
  int t;
} TVertex;

typedef struct __blake2b_param {
  uint8_t digest_length;
  uint8_t key_length;
  uint8_t fanout;
  uint8_t depth;
  uint32_t leaf_length;
  uint64_t node_offset;
  uint8_t node_depth;
  uint8_t inner_length;
  uint8_t reserved[14];
  uint8_t salt[64];
  uint8_t personal[64];
} blake2b_param;

typedef struct s_sphere {
  float3 pos;
  float r;
} t_sphere;

typedef struct vc_tmp {
  unsigned int ipad[16];
  unsigned int opad[16];

  unsigned int dgst[64];
  unsigned int out[64];

  unsigned int pim_key[64];
  int pim;

} vc_tmp_t;

typedef struct {
  float table[3];
} TempStruct;

typedef struct {
  int x;
  int y;
  float scale;
} lbp_task;

typedef struct {
  float trans_x;
  float trans_y;
  float trans_z;
  float roll;
  float pitch;
  float yaw;
} ExtrinsicParameter;

typedef struct {
  uchar perm[256];
  uchar index1;
  uchar index2;
} rc4_state_t;

typedef struct {
  float4 t0stack[1024];
  float4 t1stack[1024];
  int cstack[1024];
  unsigned int ostack[1024];
} CastStack;

typedef struct {

  Vector fixedDir;
  float4 p;
  Normal geometryN;
  Normal interpolatedN;
  Normal shadeN;

  UV defaultUV;

  Vector dpdu, dpdv;
  Normal dndu, dndv;

  unsigned int meshIndex;
  unsigned int triangleIndex;
  float triangleBariCoord1, triangleBariCoord2;

  float passThroughEvent;

  Transform localToWorld;

  unsigned int interiorVolumeIndex, exteriorVolumeIndex;

  unsigned int interiorIorTexIndex, exteriorIorTexIndex;

  unsigned int objectID;

  int intoObject, throughShadowTransparency;
} HitPoint;

typedef struct {
  int m_valInt0;
  int m_valInt1;
  int m_valInt2;
  int m_valInt3;

  float m_val0;
  float m_val1;
  float m_val2;
  float m_val3;
} SolverDebugInfo;

typedef struct {

  float3 v0;

  float3 e1;

  float3 e2;
} triangle_t;

typedef struct krb5pa {
  unsigned int user[16];
  unsigned int realm[16];
  unsigned int salt[32];
  unsigned int timestamp[16];
  unsigned int checksum[4];

} krb5pa_t;

typedef struct {
  double16 lo, hi;
} v2double16;

typedef struct {
  double x, y, z;
} dvector_t;

typedef struct int3_pair {
  int3 dx;
  int3 dy;
} int3_pair;

typedef struct {
  float x;
  float y;
  float z;

  float dx;
  float dy;
  float dz;

  unsigned int weight;
  int layer;
  unsigned short bulkpos;
  unsigned int step;
  unsigned long tof;
} PhotonStruct;

typedef struct {
  PhotonStruct *p;

  unsigned long long *x;
  unsigned int *a;

  unsigned int *thread_active;

  unsigned long long *num_terminated_photons;

  unsigned long long *Rd_xy;
  unsigned long long *Tt_xy;
  unsigned long long *fhd;

  short *bulk_info;

  unsigned long long *time_xyt;

  float *tdet_pos_x;
  float *tdet_pos_y;
} MemStruct;

typedef struct {
  float sin;
  float cos;
} rotation_t;

typedef struct {
  float2 pos;
  rotation_t rot;
} transform_t;

typedef struct {
  unsigned int texMapIndex;
  float shiftU, shiftV;
  float scaleU, scaleV;
  float scale;
} BumpMapInstance;

typedef struct {
  float value;
} ConstFloatParam;

typedef struct __store_FindData storeFindData;
enum __rtl_DigestAlgorithm {
  rtl_Digest_AlgorithmMD2,
  rtl_Digest_AlgorithmMD5,
  rtl_Digest_AlgorithmSHA,
  rtl_Digest_AlgorithmSHA1,

  rtl_Digest_AlgorithmHMAC_MD5,
  rtl_Digest_AlgorithmHMAC_SHA1,

  rtl_Digest_AlgorithmInvalid,
  rtl_Digest_Algorithm_FORCE_EQUAL_SIZE = 64
};

typedef struct UpresVarNames {
  const char *result;

  const char *ld;
  const char *startRow;
  const char *startCol;
  const char *nrRows;
  const char *nrCols;
  const char *cachedName;
} UpresVarNames;

typedef struct {
  cl_ulong surface;
  cl_ulong v0;
  cl_ulong v1;
  cl_ulong v2;
} _Triangle_unalign;

typedef struct {
  float4 m_row[3];
} Matrix3x3;

typedef struct {
  uint8_t key[32];
  uint8_t enckey[32];
  uint8_t deckey[32];
} aes256_context;

typedef struct half10 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
} half10;

typedef struct {
  union {
    int2 vecInt;
    int ints[2];
  };
} Int2CastType;

typedef struct b2clWheelJointData {
  float frequencyHz;
  float dampingRatio;

  float localAnchorA[2];
  float localAnchorB[2];
  float localXAxisA[2];
  float localYAxisA[2];

  float maxMotorTorque;
  float motorSpeed;
  int enableMotor;

  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;

  float ax[2], ay[2];
  float sAx, sBx;
  float sAy, sBy;

  float mass;
  float motorMass;
  float springMass;

  float bias;
  float gamma;
} b2clWheelJointData;

typedef struct _PAPhysicsVortexData {
  float strength, noise;
  PAPhysicsFalloffData falloff;
} PAPhysicsVortexData;

typedef struct half15 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
  float se;
  float sf;
} half15;

typedef struct _ShapeData {
  float4 m0;
  float4 m1;
  float4 m2;
  float4 m3;
  float4 linearvelocity;
  float4 angularvelocity;
  int id;
  int bvhidx;
  int mask;
  int padding1;
} ShapeData;

typedef struct b2clManifold {
  float2 localNormal;
  float2 localPoint;
  b2clManifoldPoint points[2];
  int type;
  int pointCount;

} b2clManifold;

typedef struct fermatTemp {
  mpzcl_t mpzM;
  mpzcl_t mpzE;
  mpzcl_t mpzR;
  mpzcl_t mpzHalfR;
  mpzcl_t mpzInv;
  mpzcl_t mpzOne;
  mpzcl_t mpzH;

  mpzcl32_t mpzResult;
  mpzcl32_t mpzBase;
  mpzcl32_t mpzH32;
  mpzcl32_t mpzM32;
  mpzcl32_t mpzV32;
  mpzcl32_t mpzMonProT;
  mpzcl32_t mpzOne32;

  mpzcl_t mpzXbinU;
  mpzcl_t mpzXbinV;
  mpzcl_t mpzXbinTemp;
  mpzcl_t mpzXbinTemp2;
  mpzcl_t mpzXbinTemp3;

  mpzcl_t mpzSqMod;
  mpzcll_t mpzSq;

  mpzclel_t mpzBarrettT1;
  mpzclel_t mpzBarrettT2;
  mpzclel_t mpzBarrettT3;
  mpzclel_t mpzBarrettT4;
  mpzcl_t mpzBarrettN;
  mpzcl_t mpzBarrettA;
  mpzcl_t mpzBarrettM;

  mpzcl_t mpzNewtonDen;
  mpzcl_t mpzNewtonX1;
  mpzcl_t mpzNewtonX2;
  mpzcl_t mpzNewtonX1s;
  mpzcl_t mpzNewtonX2s;
  mpzcl_t mpzNewtonX;
  mpzclel_t mpzNewtonDenP;
  mpzcl_t mpzNewtonDenPS;
  mpzcl_t mpzNewton2;
  mpzcl_t mpzNewton2s;
  mpzcl_t mpzNewtonDiff;
  mpzclel_t mpzNewtonProd;

  mpzcl_t mpzOriginMinusOne;
  mpzcl_t mpzOriginPlusOne;
  mpzcl_t mpzFixedFactor;
  mpzcl_t mpzChainOrigin;
  mpzcl_t mpzOriginShift;

  uint32_t i[1024];
  uint32_t e[1024];
  uint32_t p[1024];
  uint64_t al[1024];
  uint64_t bl[1024];
  uint32_t il[1024];
} fermatTemp_t;

typedef struct {
  float M[3][3];
} ago_perspective_matrix_t;

typedef struct _avg_nodestats {
  int counter;
  double boltzmann_factor, boltzmann_factor_sq;
  double acceptance_rate;
  double acceptance_rate_insert, acceptance_rate_remove,
      acceptance_rate_displace, acceptance_rate_ptemp;
  double acceptance_rate_adiabatic, acceptance_rate_spinflip,
      acceptance_rate_volume, acceprance_rate_ptemp;
  double cavity_bias_probability, cavity_bias_probability_sq;
  double polarization_iterations, polarization_iterations_sq;
} avg_nodestats_t;

typedef struct {
  int fixtureAIndex;
  int fixtureBIndex;
  int bodyAIndex;
  int bodyBIndex;
} clb2SDContact;

typedef struct {
  int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_dswap_params_t;

typedef struct {
  __global float *learningInputs, *learningOutputs;
  int totalLearningLines;
  __global float *testInputs, *testOutputs;
  int totalTestLines;
} taskData_t;

typedef struct _antinfo {
  int numberColony;
  int numberAgentPerColony;
  float highRanking;
  float evaporatePeromone;
  float c1, c2, c3, c4;
} antinfo;

typedef struct {
  float4 origx, origy, origz;
  float4 edge1x, edge1y, edge1z;
  float4 edge2x, edge2y, edge2z;
  uint4 primitives;
} QuadTiangle;

typedef struct rakp {
  unsigned int salt_buf[128];
  unsigned int salt_len;

} rakp_t;

typedef struct {
  bool manager_lock;
  bool user_lock;
  size_t bytes;
} locked_info;

typedef struct {
  int field;
} FStruct;

typedef struct {
  FStruct *fstruct;
} SStruct;

typedef struct {
  float level_gr;
  float level_r;
  float level_b;
  float level_gb;
  unsigned int color_bits;
} CLBLCConfig;

typedef struct _cl_body_t {
  double4 m_position;
  double4 m_velocity;
  double4 m_acceleration;
  double m_mass;
  double m_radius;
} cl_body_t;

typedef struct {
  CRTypedef rstruct;
} CPRTypedef;

typedef struct {
  unsigned int length;
  unsigned int iterations;
  uchar salt[256];
  uint32_t crypto_size;
  uchar ct[64 / 2];
} keyring_salt;

typedef struct {
  float gain;
  float threshold;
  float log_min;
  float log_max;
  float width;
  float height;
} CLRetinexConfig;

typedef struct {

  float8 position;
  float8 dimensions;

  unsigned int depth;
  unsigned int child_indices[9];
  int parent_index;
  unsigned int sibling_index;
  unsigned int leaf_count;
  unsigned int leaf_index;

  unsigned char has_children;

  node_value_t value;

} node_t;

typedef struct {
  void *mtx;
  int data;
} my_args_t;

typedef struct b2clDistanceJointDatastruct {
  float frequencyHz;
  float dampingRatio;
  float bias;

  float localAnchorA[2];
  float localAnchorB[2];
  float gamma;
  float nlength;

  float u[2];
  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  float mass;
} b2clDistanceJointData;

typedef struct b3ConvexPolyhedronData b3ConvexPolyhedronData_t;

struct b3ConvexPolyhedronData {
  float4 m_localCenter;
  float4 m_extents;
  float4 mC;
  float4 mE;

  float m_radius;
  int m_faceOffset;
  int m_numFaces;
  int m_numVertices;

  int m_vertexOffset;
  int m_uniqueEdgesOffset;
  int m_numUniqueEdges;
  int m_unused;
};

typedef struct {
  float H;
  float u;
  float v;
} CFluxVec;

typedef struct {
  uint8_t length;
  uint8_t v[55];
} pass_t;

typedef struct _event_list *event_list;

struct _event_list {
  cl_event event;
  event_list next;
};

typedef struct {
  float rng0, rng1;
  unsigned int pixelIndex, pass;

  Spectrum radiance;
} SobolSampleWithoutAlphaChannel;

typedef struct {

  DotVector v[(3 + 5 - 1)];
} DotFeatureVector;

typedef struct sha384_hmac_ctx {
  sha384_ctx_t ipad;
  sha384_ctx_t opad;

} sha384_hmac_ctx_t;

typedef struct {
  unsigned int width, height;
  int superSamplingSize;
  int activateFastRendering;
  int enableShadow;

  unsigned int maxIterations;
  float epsilon;
  float mu[4];
  float light[3];
  Camera camera;
} RenderingConfig;

typedef struct {
  signed char decimation_mode;
  signed char quantization_mode;
  signed char is_dual_plane;
  signed char permit_encode;
  signed char permit_decode;
  float percentile;
} t_block_mode;

typedef struct {
  int decimation_mode_count;
  int decimation_mode_samples[87];
  int decimation_mode_maxprec_1plane[87];
  int decimation_mode_maxprec_2planes[87];
  float decimation_mode_percentile[87];
  int permit_encode[87];
  decimation_table decimation_tables[87 + 1];
  t_block_mode block_modes[2048];

  int texelcount_for_bitmap_partitioning;
  int texels_for_bitmap_partitioning[64];
} block_size_descriptor;

typedef struct nei_str {

  int x, y, z;
  int number;
  long offset;

} nei_str;

typedef struct {
  constant char *p3;
  global char *p4;
  char *p5;
} StructTy2;

typedef struct __attribute__((aligned(4))) {

  float uDir[2];
} EDFSample;

typedef struct {
  float a;
  float b;
  float c;
} Structure;

typedef struct device_s {
  cl_platform_id _platform;
  cl_device_id _device_id;
  cl_context _context;
  cl_command_queue _queue;
} device_t;

typedef struct SPHParams {
  float rest_density;
  float mass;
  float rest_distance;
  float smoothing_distance;
  float simulation_scale;

  float boundary_stiffness;
  float boundary_dampening;
  float boundary_distance;
  float K;

  float viscosity;
  float velocity_limit;
  float xsph_factor;

  float friction_coef;
  float restitution_coef;
  float shear;
  float attraction;
  float spring;

  float EPSILON;

  float wpoly6_coef;
  float wpoly6_d_coef;
  float wpoly6_dd_coef;
  float wspiky_coef;
  float wspiky_d_coef;

  float wspiky_dd_coef;
  float wvisc_coef;
  float wvisc_d_coef;
  float wvisc_dd_coef;

  int num;
  int max_num;

  float4 gravity;

} SPHParams;

typedef struct {
  float x;
  float y;
  float z;
} FLOAT3;

typedef struct saph_sha1_tmp {
  unsigned int digest_buf[5];

} saph_sha1_tmp_t;

typedef struct TfLiteComplex64 {
  float re, im;
} TfLiteComplex64;

typedef struct _d_point {
  double x;
  double y;
} d_point_t;

typedef struct _d_subsampling_kernel_info {
  d_point_t point;
  double xbin;
  double ybin;
  double val;
} d_ss_kinfo_t;

typedef struct {
  float4 m_pos;
  float4 m_quat;
  float4 m_linVel;
  float4 m_angVel;

  unsigned int m_collidableIdx;
  float m_invMass;
  float m_restituitionCoeff;
  float m_frictionCoeff;
} BodyData;

typedef struct shaOutput {
  uint32_t mod[1024];
  mpzcl_t mpzHash;
  uint32_t nonce[1024];
} shaOutput_t;

typedef struct {
  unsigned int cs_buf[0x100];
  unsigned int cs_len;

} cs_t;

typedef struct __attribute__((packed)) _chan3_t {
  uchar ctrl;
  unsigned int count;
  data_type wgtCent;
  int sum_sq;
  unsigned char search_idx;
} chan3_t;

typedef struct {
  long long ll[6];
} XXH32_state_t;

typedef struct TfLiteNode {

  TfLiteIntArray *inputs;

  TfLiteIntArray *outputs;

  TfLiteIntArray *intermediates;

  TfLiteIntArray *temporaries;

  void *user_data;

  void *builtin_data;

  const void *custom_initial_data;
  int custom_initial_data_size;

  struct TfLiteDelegate *delegate;
} TfLiteNode;

typedef struct {
  lane_data wvec[4];
} data_wng_ddr;

typedef struct intintIdentityPerfectCLHash_Bucket {
  int key;
  int value;
} intintIdentityPerfectCLHash_Bucket;

typedef struct _surf_preserve_rotation {
  double alpha1, alpha2, beta1, beta2, gamma1, gamma2;
} surf_preserve_rotation;

typedef struct _vdw {
  char mtype[512];
  double energy;
  struct _vdw *next;
} vdw_t;

typedef struct grid_t {
  double dt;
  double dx, dy, dz;
  int ai, aj, ak;
  int ni, nj, nk;
} grid_t;

typedef struct _nodestats {
  int accept, reject;
  int accept_insert, accept_remove, accept_displace, accept_adiabatic,
      accept_spinflip, accept_volume, accept_ptemp;
  int reject_insert, reject_remove, reject_displace, reject_adiabatic,
      reject_spinflip, reject_volume, reject_ptemp;
  double boltzmann_factor;
  double acceptance_rate;
  double acceptance_rate_insert, acceptance_rate_remove,
      acceptance_rate_displace;
  double acceptance_rate_adiabatic, acceptance_rate_spinflip,
      acceptance_rate_volume, acceptance_rate_ptemp;
  double cavity_bias_probability;
  double polarization_iterations;
} nodestats_t;

typedef struct _sorbateInfo {
  char id[16];
  double mass;
  int currN;
  double percent_wt;
  double percent_wt_me;
  double excess_ratio;
  double pore_density;
  double density;
} sorbateInfo_t;

typedef struct _sorbateAverages {
  double avgN, avgN_sq, avgN_err;
  double percent_wt, percent_wt_sq, percent_wt_err;
  double percent_wt_me, percent_wt_me_sq, percent_wt_me_err;
  double excess_ratio, excess_ratio_sq, excess_ratio_err;
  double pore_density, pore_density_sq, pore_density_err;
  double density, density_sq, density_err;
  double selectivity, selectivity_err;
  struct _sorbateAverages *next;
} sorbateAverages_t;

typedef struct _checkpoint {
  int movetype, biased_move;
  int thole_N_atom;
  molecule_t *molecule_backup, *molecule_altered;
  molecule_t *head, *tail;
  observables_t *observables;
} checkpoint_t;

typedef struct _file_pointers {
  FILE *fp_energy;
  FILE *fp_energy_csv;
  FILE *fp_xyz;
  FILE *fp_field;
  FILE *fp_histogram;
  FILE *fp_frozen;

  FILE *fp_traj_replay;
  FILE *fp_surf;
} file_pointers_t;

typedef struct _system {
  int ensemble;
  int gwp;
  int cuda;
  int opencl;

  int preset_seeds_on;
  uint32_t preset_seeds;
  int rng_initialized;

  int parallel_restarts;

  int surf_fit_multi_configs;
  char *multi_fit_input;
  int surf_fit_arbitrary_configs;
  int surf_qshift_on, surf_scale_epsilon_on, surf_scale_r_on,
      surf_scale_omega_on, surf_scale_sigma_on, surf_scale_q_on,
      surf_scale_pol_on;
  int surf_weight_constant_on, surf_global_axis_on, surf_descent,
      surf_scale_alpha_on, surf_scale_c6_on, surf_scale_c8_on,
      surf_scale_c10_on;
  fileNode_t fit_input_list;
  double surf_scale_epsilon, surf_scale_r, surf_scale_omega, surf_scale_sigma,
      surf_scale_q, surf_scale_alpha, surf_scale_pol, surf_scale_c6,
      surf_scale_c8, surf_scale_c10;
  double surf_quadrupole, surf_weight_constant;
  double fit_start_temp, fit_max_energy, fit_schedule;
  int fit_boltzmann_weight;
  double fit_best_square_error;
  char **surf_do_not_fit_list;

  int surf_preserve, surf_decomp;
  double surf_min, surf_max, surf_inc, surf_ang;
  surf_preserve_rotation *surf_preserve_rotation_on;
  int surf_virial;
  char *virial_output;
  double *virial_coef;
  double virial_tmin, virial_tmax, virial_dt;
  int virial_npts;
  int ee_local;
  double range_eps, range_sig, step_eps, step_sig;

  int numsteps, corrtime, step;
  int ptemp_freq;
  double move_factor, rot_factor, insert_probability;
  double adiabatic_probability, spinflip_probability, gwp_probability,
      volume_probability;
  double volume_change_factor, last_volume;

  int cavity_autoreject_absolute;
  int count_autorejects;

  int parallel_tempering;
  double max_temperature;
  ptemp_t *ptemp;

  int cavity_bias, cavity_grid_size;
  cavity_t ***cavity_grid;
  int cavities_open;
  double cavity_radius, cavity_volume, cavity_autoreject_scale,
      cavity_autoreject_repulsion;

  int spectre;
  double spectre_max_charge, spectre_max_target;

  int simulated_annealing, simulated_annealing_linear;
  double simulated_annealing_schedule;
  double simulated_annealing_target;

  int rd_only, rd_anharmonic;
  double rd_anharmonic_k, rd_anharmonic_g;
  int sg, dreiding, waldmanhagler, lj_buffered_14_7, halgren_mixing, c6_mixing,
      disp_expansion;
  int extrapolate_disp_coeffs, damp_dispersion, disp_expansion_mbvdw;
  int axilrod_teller, midzuno_kihara_approx;

  int wolf;
  double ewald_alpha, polar_ewald_alpha;
  int ewald_alpha_set, polar_ewald_alpha_set;
  int ewald_kmax;

  int polarization, polarvdw, polarizability_tensor;
  int cdvdw_exp_repulsion, cdvdw_sig_repulsion, cdvdw_9th_repulsion;
  int iter_success;
  int polar_iterative, polar_ewald, polar_ewald_full, polar_zodid, polar_palmo,
      polar_rrms;
  int polar_gs, polar_gs_ranked, polar_sor, polar_esor, polar_max_iter,
      polar_wolf, polar_wolf_full, polar_wolf_alpha_lookup;
  double polar_wolf_alpha, polar_gamma, polar_damp, field_damp, polar_precision;
  int damp_type;
  double **A_matrix, **B_matrix, C_matrix[3][3];
  vdw_t *vdw_eiso_info;
  double *polar_wolf_alpha_table, polar_wolf_alpha_lookup_cutoff;
  int polar_wolf_alpha_table_max;

  int feynman_hibbs, feynman_kleinert, feynman_hibbs_order;
  int vdw_fh_2be;
  int rd_lrc, rd_crystal, rd_crystal_order;

  int h2_fugacity, co2_fugacity, ch4_fugacity, n2_fugacity, user_fugacities;

  int wrapall;
  char *job_name;
  char *pqr_input, *pqr_output, *pqr_restart, *traj_input, *traj_output,
      *energy_output, *energy_output_csv, *surf_output, *xyz_output;
  int read_pqr_box_on;
  int long_output;
  int surf_print_level;
  char *dipole_output, *field_output, *histogram_output, *frozen_output;
  char *insert_input;
  double max_bondlength;

  int num_insertion_molecules;
  molecule_t *insertion_molecules;
  molecule_t **insertion_molecules_array;

  int quantum_rotation, quantum_rotation_hindered;
  double quantum_rotation_B;
  double quantum_rotation_hindered_barrier;
  int quantum_rotation_level_max, quantum_rotation_l_max,
      quantum_rotation_theta_max, quantum_rotation_phi_max,
      quantum_rotation_sum;
  int quantum_vibration;

  grid_t *grids;
  int calc_hist;
  double hist_resolution;
  int n_histogram_bins;

  double temperature, pressure, free_volume, total_energy, N;
  double *fugacities;
  int fugacitiesCount;

  int natoms;
  atom_t **atom_array;
  molecule_t **molecule_array;

  int calc_pressure;
  double calc_pressure_dv;

  double scale_charge;
  int independent_particle;

  pbc_t *pbc;
  molecule_t *molecules;

  nodestats_t *nodestats;
  avg_nodestats_t *avg_nodestats;
  observables_t *observables;
  avg_observables_t *avg_observables;

  int sorbateCount;
  sorbateInfo_t *sorbateInfo;
  int sorbateInsert;
  sorbateAverages_t *sorbateGlobal;

  checkpoint_t *checkpoint;
  file_pointers_t file_pointers;

} system_t;

typedef struct {
  unsigned int indices[3];
  unsigned int mat_index;
} TriangleData;

typedef struct {
  unsigned int cracked;
} keyring_hash;

typedef struct PRNG {

  ulong4 state;

  unsigned int pointer;

  ulong4 seed;
} PRNG;

typedef struct {
  int offs[4];
  int strds[4];
  char isSeq[4];
} IndexKernelParam_t;

typedef struct InternalKeypoint {
  float x;
  float y;
  float tracking_status;
  float dummy;
} InternalKeypoint;

typedef struct {
  unsigned int bucketIndex, pixelOffset, passOffset, pass;
} RandomSample;

typedef struct {
  int type;
  int index;
  float separation;
} b2clEPAxis;

typedef struct {
  float4 m_aabbMin;
  float4 m_aabbMax;
  float4 m_quantization;
  int m_numNodes;
  int m_numSubTrees;
  int m_nodeOffset;
  int m_subTreeOffset;

} b3BvhInfo;

typedef struct {
  float4 direction;
  float4 index_to_physical_point;
  float4 physical_point_to_index;
  float2 spacing;
  float2 origin;
  uint2 size;
} GPUImageBase2D;

typedef struct {
  uint32_t rounds;
  uint32_t length;
  uint32_t final;
  buffer_64 salt[(16 / 8)];
} sha512_salt;

typedef struct {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_rootNodeIndex;

  int m_subtreeSize;
  int m_padding[3];
} btBvhSubtreeInfo;

typedef struct dst_local_bucket {
  element_t data[128];
} dst_local_bucket_t;

typedef struct {
  unsigned int array[1];
} boundary_index_array_1;

typedef struct __attribute__((packed)) {
  int int_value;
  float float_value;
} data_struct;

typedef struct {
  cl_float4 vec;
  float input;
} UserObj;

typedef struct float0 {
  float s0;
} float0;

typedef struct _shl1_data_t {
  float4 coeff_r, coeff_g, coeff_b;
} shl1_data_t;

typedef struct {
  unsigned int width;
  unsigned int height;
  float cx;
  float cy;
  float fx;
  float fy;
  float fov;
  float skew;
  float c;
  float d;
  float e;
  unsigned int poly_length;
  float poly_coeff[16];
  int flip;
} IntrinsicParameter;

typedef struct {
  unsigned int triangle;
  float distance_squared;
} triangle_distance_pair;

typedef struct {
  unsigned int pixelX, pixelY;
  float filmX, filmY;

  Spectrum radiancePerPixelNormalized[8];
  float alpha;
  float depth;
  float4 position;
  Normal geometryNormal;
  Normal shadingNormal;

  unsigned int materialID;

  unsigned int objectID;
  Spectrum directDiffuse, directDiffuseReflect, directDiffuseTransmit;
  Spectrum directGlossy, directGlossyReflect, directGlossyTransmit;
  Spectrum emission;
  Spectrum indirectDiffuse, indirectDiffuseReflect, indirectDiffuseTransmit;
  Spectrum indirectGlossy, indirectGlossyReflect, indirectGlossyTransmit;
  Spectrum indirectSpecular, indirectSpecularReflect, indirectSpecularTransmit;
  float directShadowMask;
  float indirectShadowMask;
  UV uv;
  float rayCount;
  Spectrum irradiance, irradiancePathThroughput;
  Spectrum albedo;

  int firstPathVertexEvent;
  int isHoldout;

  int firstPathVertex, lastPathVertex;
} SampleResult;

typedef struct {
  float totalI;

  unsigned int largeMutationCount, smallMutationCount;
  unsigned int current, proposed, consecutiveRejects;

  float weight;

  SampleResult currentResult;
} MetropolisSample;

typedef struct {
  float rgb_power;
  float rgb_base_weight;
  float rgb_mean_weight;
  float rgb_stdev_weight;
  float alpha_power;
  float alpha_base_weight;
  float alpha_mean_weight;
  float alpha_stdev_weight;
  float rgb_mean_and_stdev_mixing;
  int mean_stdev_radius;
  int enable_rgb_scale_with_alpha;
  int alpha_radius;
  int ra_normal_angular_scale;
  float block_artifact_suppression;
  float rgba_weights[4];

  float block_artifact_suppression_expanded[216];

  int partition_search_limit;
  float block_mode_cutoff;
  float texel_avg_error_limit;
  float partition_1_to_2_limit;
  float lowest_correlation_cutoff;
  int max_refinement_iters;
} error_weighting_params;

typedef struct __attribute__((aligned(2))) {
  DDFHead ddfHead;
  uchar numEnvEEDFs __attribute__((aligned(2)));
  ushort offsetsEnvEEDFs[4] __attribute__((aligned(2)));
} EnvEDFHead;

typedef struct {
  unsigned int meshIndex;

  unsigned int bvhRoot;

  unsigned int _reserved1;
  unsigned int _reserved2;

  float4 transformMat0;
  float4 transformMat1;
  float4 transformMat2;
  float4 transformMat3;
} MeshInstance;

typedef struct agilekey_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[5];
  unsigned int out[5];

} agilekey_tmp_t;

typedef struct Particle {
  float4 pos;
  float4 float3;
  float4 vel;
  float4 velChange;
  float4 colorChange;
  float sizeChange;
  float life;
  float size;
} Particle;

typedef struct {
  int m_constraintType;
  int m_rbA;
  int m_rbB;
  float m_breakingImpulseThreshold;

  float4 m_pivotInA;
  float4 m_pivotInB;
  float4 m_relTargetAB;

  int m_flags;
  int m_padding[3];
} b3GpuGenericConstraint;

typedef struct {
  int num_simbolos;
  type_simbolo simbolos[50];
} t_escolha;

typedef struct s_shader {
  float3 col1;
  float3 col2;
  float3 col3;
} t_shader;

typedef struct {
  unsigned int cracked;
} blockchain_out;

typedef struct {
  float4 _0;
  float4 _1;
} Tuple_float4_float4;

typedef struct {
  Tuple_float4_float4 _0;
  float4 _1;
} Tuple_Tuple_float4_float4_float4;

typedef struct {
  Matrix3x3 m_invInertiaWorld;
  Matrix3x3 m_initInvInertia;
} BodyInertia;

typedef struct Coordinates2D {
  int x;
  int y;
} Coordinates2D;

typedef struct {
  float poly_6;
  float poly_6_gradient;
  float poly_6_laplacian;
  float spiky;
  float viscosity;
} precomputed_kernel_values;

typedef struct {
  unsigned int ctx_a;
  unsigned int ctx_b;
  unsigned int ctx_c;
  unsigned int ctx_d;
  unsigned int ctx_e;
  unsigned int ctx_f;
  unsigned int ctx_g;
  unsigned int ctx_h;
  unsigned int cty_a;
  unsigned int cty_b;
  unsigned int cty_c;
  unsigned int cty_d;
  unsigned int cty_e;
  unsigned int cty_f;
  unsigned int cty_g;
  unsigned int cty_h;
  unsigned int merkle;
  unsigned int ntime;
  unsigned int nbits;
  unsigned int nonce;
  unsigned int fW0;
  unsigned int fW1;
  unsigned int fW2;
  unsigned int fW3;
  unsigned int fW15;
  unsigned int fW01r;
  unsigned int fcty_e;
  unsigned int fcty_e2;
} dev_blk_ctx;

typedef struct def_ParticleData {
  float density;

  float3 force;
  float color_field;
} ParticleData;

typedef struct {
  int activity;
  global Segment *segment;
  int segmentIdx;
} BestMatchingSegmentStruct;

typedef struct {
  unsigned int texIndex;
} AbsTexParam;

typedef struct mutationData {
  char mods[16];
  unsigned int locs[16];
  int ins[16];

  int len;
} mData;

typedef struct {
  int x;
  int y;
} Edge;

typedef struct {
  float data[64 / 2];
} CHAN_WIDTH;

typedef struct {
  const global float4 *intersectFirst;
  const global float4 *intersectSecond;
  const global float4 *normalsFirst;
  const global float4 *normalsSecond;
  const global float4 *textureFirst;
} samplesData_t;

typedef enum {
  NrmoptionNoArg,
  NrmoptionIsArg,
  NrmoptionStickyArg,
  NrmoptionSepArg,
  NrmoptionResArg,
  NrmoptionSkipArg,
  NrmoptionSkipLine,
  NrmoptionSkipNArgs

} NrmOptionKind;

typedef struct {
  float dd;
  float dm;
  float dl;
  float md;
  float mm;
  float ml;
  float ld;
  float lm;
  float ll;
} bandwidth_t;

typedef struct half5 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
} half5;

typedef struct _MonteCalroAttrib {
  float4 strikePrice;
  float4 c1;
  float4 c2;
  float4 c3;
  float4 initPrice;
  float4 sigma;
  float4 timeStep;
} MonteCarloAttrib;

typedef struct quat {
  float x, y, z, w;
} quat;

typedef struct tag_rendering_params {
  float4 backgroundColor;

  float3 modelScale;
  unsigned int illumType;

  unsigned int imgEss;
  unsigned int showEss;
  unsigned int useLinear;
  unsigned int useGradient;

  unsigned int technique;
  unsigned int seed;
  unsigned int iteration;
} rendering_params;

typedef struct struct_char_x3 {
  char x, y, z;
} struct_char_x3;

typedef struct half6 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
} half6;

typedef struct {
  global Cell *cell;
  int cellIdx;
  global Segment *segment;
  int segmentIdx;
} BestMatchingCellStruct;

typedef struct {
  union {
    float4 m_min;
    float m_minElems[4];
    int m_minIndices[4];
  };
  union {
    float4 m_max;
    float m_maxElems[4];
    int m_maxIndices[4];
  };
} btAabbCL;

typedef struct _kdTree_t {

  unsigned int count;
  data_type wgtCent;
  int sum_sq;
  data_type bnd_lo;
  data_type bnd_hi;
  unsigned int idx;
  unsigned int left;
  unsigned int right;
} kdTree_t;

typedef struct __attribute__((packed)) _chan1_t {
  uchar ctrl;
  unsigned int u;
  kdTree_t tn;
  unsigned int cs;
  bool d;
  unsigned char current_k;
  unsigned char current_indices;
  data_type current_centers;
} chan1_t;

typedef struct {
  cl_uint index;
  size_t size;
  uint texture;
  cl_mem buffer;
} Argument;

typedef struct {
  uint8 x;
  uint8 c;
} mwc64xvec8_state_t;

typedef struct {
  unsigned int seq;
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
} SobolSampler;

typedef struct aperture aperture_t;

struct aperture {
  float ap_m, ap_h, ap_t, len;
};

typedef struct {
  TVector min, max;
} TBounds;

typedef struct {
  float3 v1;
  float3 v2;
  float3 v3;
} Trgl;

typedef struct knode2 {
  int location;
  int indices[256 + 1];
  int keys[256 + 1];
  bool is_leaf;
  int num_keys;
} knode2;

typedef struct {
  unsigned int length;
  uchar v[255];
} keyring_password;

typedef struct {
  char lane[8];
} DPTYPE_SCAL;

typedef struct {
  DPTYPE_SCAL data[(416 / 32)];
} DPTYPE_PE_SCAL;

typedef struct ethereum_scrypt {
  unsigned int salt_buf[16];
  unsigned int ciphertext[8];

} ethereum_scrypt_t;

typedef struct {
  unsigned int v[256 / 32];
} odf_out;

typedef struct {
  float c[3];
} RGBColor;

typedef struct {

  unsigned int cameraFilmWidth, cameraFilmHeight;
  unsigned int tileStartX, tileStartY;
  unsigned int tileWidth, tileHeight;
  unsigned int tilePass, aaSamples;
  unsigned int multipassIndexToRender;

} TilePathSamplerSharedData;

typedef struct {
  uint8_t data[16];
} physical_compressed_block;

typedef struct float14 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
  float sd;
} float14;

typedef struct _VshFieldMap {
  uchar offset;
  uchar shift;
  ulong mask;
  bool sign_extend;
} VshFieldMap;

typedef struct {
  int a, b, c, d;
} S;

typedef struct {
  union {
    unsigned char c[8][8][sizeof(int)];
    int v[8][8];
  } xkeys;
} opencl_lm_transfer;

typedef struct ripemd160_hmac_ctx_vector {
  ripemd160_ctx_vector_t ipad;
  ripemd160_ctx_vector_t opad;

} ripemd160_hmac_ctx_vector_t;

typedef struct b2clWeldJointData {
  float frequencyHz;
  float dampingRatio;
  float bias;

  float localAnchorA[2];
  float localAnchorB[2];
  float referenceAngle;
  float gamma;

  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  b2clMat33 mass;
} b2clWeldJointData;

typedef struct {
  float x0, y0, z0;
  float x1, y1, z1;
} RT_BBox;

typedef struct tri_context {
  int dom_axis;
  float xplane, yplane, zplane;
  float pxa, pya, dxdya, pxb, pyb, dxdyb;
  int x0p, x1p;
  int x0, x1, x2;
  int xmin, xmax;
  unsigned triangle;
  int width;
  float4 normal;
  __global unsigned char *image;

  __global unsigned *octree;
  unsigned octree_size;
} tri_context;

typedef struct DetectionWindow {
  ushort x;
  ushort y;
  ushort width;
  ushort height;
  ushort idx_class;
  float score;
} DetectionWindow;

typedef struct keychain_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} keychain_tmp_t;

typedef struct LS_t {
  float2 startCoords, endCoords;
  int startIndex, endIndex;
  int leftPtr, rightPtr;

  int startCount, endCount;
  int maxDist;
  int polyid;
  int npix;
  int level;
} LS_t;

typedef struct def_ClothTriangleData {
  unsigned int triangleID;
  int neighbourIDs[3];
  float mass;
} ClothTriangleData;

typedef struct {
  int a_field_identifier_that_is_256_characters_long_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789;
} ATypeIdentifierThatIs256CharactersLong01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567;

typedef struct {
  unsigned int m_key;
  unsigned int m_value;
} SortData;

struct coprthr_device_info {

  unsigned int max_compute_units;
  unsigned int max_freq;
  char *name;
  char *vendor;
  unsigned int vendorid;
  char *drv_version;
  char *profile;
  char *version;
  char *extensions;

  int arch_id;
  int memsup;

  size_t global_mem_sz;
  size_t local_mem_sz;

  int devsup;
};

typedef struct {
  int version;
  uint32_t iterations;
  uint8_t hash[32];
  uint8_t salt[32];
} pwsafe_salt;

typedef struct {
  int width;
  int height;
  float focus;
  float radius;
} sParamsDOF;

typedef struct KernelVarNames {
  const char *A;
  const char *B;
  const char *C;
  const char *LDS;
  const char *coordA;
  const char *coordB;
  const char *k;
  const char *skewA;
  const char *skewB;
  const char *skewK;
  const char *sizeM;
  const char *sizeN;
  const char *sizeK;
  const char *lda;
  const char *ldb;
  const char *ldc;
  const char *vectCoordA;

  const char *vectCoordB;

  const char *startM;
  const char *startN;
  const char *startK;
  const char *alpha;
  const char *beta;
} KernelVarNames;

typedef struct {
  unsigned int texIndex, brightnessTexIndex, contrastTexIndex;
} BrightContrastTexParam;

typedef struct _NhlCoord {
  float x;
  float y;
} NhlCoord;

typedef struct CvTreeNodeIterator {
  const void *node;
  int level;
  int max_level;
} CvTreeNodeIterator;

typedef struct {
  int first;
  float4 second;
  uint3 third[4];
} PrivateStruct;

typedef struct Tensor4D {
  __global uchar *ptr;
  int offset_first_element_in_bytes;
  int stride_x;
  int stride_y;
  int stride_z;
  int stride_w;
} Tensor4D;

typedef struct b2clRevoluteJointData {
  float localAnchorA[2];
  float localAnchorB[2];

  int enableMotor;
  float maxMotorTorque;
  float motorSpeed;

  int enableLimit;
  float referenceAngle;
  float lowerAngle;
  float upperAngle;

  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  b2clMat33 mass;
  float motorMass;
  int limitState;
} b2clRevoluteJointData;

typedef struct {
  float4 SourcePosition;
  float4 SourceColor;
  float4 TargetPosition;
  float4 TargetColor;
  float Width;

  unsigned int SourceID;
  unsigned int TargetID;

} EdgeInstance;

typedef struct pbkdf2_sha256_tmp {
  unsigned int ipad[8];
  unsigned int opad[8];

  unsigned int dgst[32];
  unsigned int out[32];

} pbkdf2_sha256_tmp_t;

typedef struct krb5pa_17_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[10];
  unsigned int out[10];

} krb5pa_17_tmp_t;

typedef struct WorkGroupSum_t {
  int workGroupSum;
  int inclusivePrefix;
} WorkGroupSum;

typedef struct oldoffice01 {
  unsigned int version;
  unsigned int encryptedVerifier[4];
  unsigned int encryptedVerifierHash[4];
  unsigned int rc4key[2];

} oldoffice01_t;

typedef struct oldoffice34 {
  unsigned int version;
  unsigned int encryptedVerifier[4];
  unsigned int encryptedVerifierHash[5];
  unsigned int rc4key[2];

} oldoffice34_t;

typedef struct {
  OutputWidthVector data;
  bool pe_output_relu;
  bool is_QVECTOR;
} PeOutput;

typedef struct Params {
  float neighbourRadiusSqr;

  float maxSteeringForce;
  float maxBoidSpeed;

  float wanderRadius;
  float wanderJitter;
  float wanderDistance;
  float wanderWeight;

  float separationWeight;
  float cohesionWeight;
  float alignmentWeight;

  unsigned int boidCount;
} Params;

typedef struct _ray_packet_t {
  float4 o, d;
  float4 c;
  float4 do_dx, dd_dx, do_dy, dd_dy;
} ray_packet_t;

typedef struct tag_raycast_params {
  float samplingRate;
  unsigned int useAO;
  unsigned int contours;
  unsigned int aerial;

  float3 brickRes;
} raycast_params;

typedef struct {
  unsigned int tCount;
  unsigned int vCount;
  unsigned int nCount;
  float radius;
  float3 center;
} MeshInfo;

typedef struct PathInfo {

  HitInfo hit_info;
  float4 dir;
  float4 contrib;
  float fwd_pdf;
  float rev_pdf;

} PathInfo;

typedef struct ConvolutionalLayer {
  int numberOfFilters;
  Filter filters[20];
} ConvolutionalLayer;

typedef struct uctx_t {

  cl_ulong h[8];

  cl_ulong t[2];
} uctx_t;

typedef struct {
  unsigned int rngPass;
  float rng0, rng1;

  unsigned int pass;
} TilePathSample;

typedef struct {
  float8 lo, hi;
} v2float8;

typedef struct {
  union {
    unsigned int p;
    struct {
      uchar p1, p2, p3;
      uchar bf;
    };
  };
} OTnode;

typedef struct ge25519_niels_t {
  unsigned int ysubx, xaddy, t2d;
} ge25519_niels;

typedef struct outer {
  inner x;
} outer;

typedef struct KernelGlobals {
  __constant KernelData *data;

} KernelGlobals;

typedef struct __attribute__((aligned(1))) {
  FresnelHead head;
} FresnelNoOp;

typedef struct {
  TBounds extent;
  int elements_num, elements_allocated;
  TBounds *elements;
  bool tree_built;
  int tree_size;
  TBounds *tree_safety_boxes;
  int *tree_link;
} TKDTree;

typedef struct {
  float x;
  float y;
} Centroid;

typedef struct par_str {

  float alpha;

} par_str;

typedef struct {
  float4 m_row[3];
} Matrix3x3f;

typedef struct {
  int maxEpochs;
  float minProgress;
  float maxGeneralizationLoss;
} criteria_t;

typedef struct {
  setup_t setup;
  criteria_t criteria;
  uint16 state;
  float currentSquareErrorCounter;
  float bestSquareError[2];
  __global float *squareErrorHistory;
} neuralNetwork_t;

typedef struct __attribute__((aligned(4))) {
  DistributionHead head;
  unsigned int numItems __attribute__((aligned(4)));
  int offsetPMF;
  int offsetCDF;
} Discrete1D;

typedef struct {
  Mueller R;
  Mueller G;
  Mueller B;
} Modificator;

typedef struct LodePNGDecompressSettings LodePNGDecompressSettings;
struct LodePNGDecompressSettings {
  unsigned ignore_adler32;

  unsigned *custom_zlib;

  unsigned *custom_inflate;

  const void *custom_context;
};

typedef struct {
  float h;
  float h2;
  float stiffness;
  float mass;
  float mass2;
  float density0;
  float viscosity;
  float surfaceTension;

  float CWd;
  float CgradWd;
  float CgradWp;
  float ClaplacianWv;
  float CgradWc;
  float ClaplacianWc;
} GPUSPHFluid;

typedef struct {
  unsigned int frontMatIndex;
  unsigned int backMatIndex;
} TwoSidedParam;

typedef struct {
  ProjectiveCamera projCamera;
} OrthographicCamera;

typedef struct aes_ctx {
  uint32_t sk[60];
  uint32_t sk_exp[120];

  unsigned int num_rounds;
} AES_CTX;

typedef struct {
  float4 v[4];
} mat4t;

typedef struct chacha20 {
  unsigned int iv[2];
  unsigned int plain[2];
  unsigned int position[2];
  unsigned int offset;

} chacha20_t;

typedef struct {
  float shiftU, shiftV;
  Spectrum gain;
  unsigned int width, height;
} InfiniteLight;

typedef struct {
  float ee_gain;
  float ee_threshold;
  float nr_gain;
} CLEeConfig;

typedef struct __attribute__((aligned(4))) Stump {
  float4 st __attribute__((aligned(4)));
} Stump;

typedef struct s_img_info {
  ulong index;
  int2 size;
} t_img_info;

typedef struct {
  double x;
  double y;
  double z;
} XYZ;

typedef struct {
  float z;
  int i;
} sSortZ;

typedef struct {
  float time;
  float tau;
} CFlux2DCalculatorPars;

typedef struct {
  unsigned int v[4];
} crypt_md5_hash;

typedef struct {
  float *buf;
  int dim[2];
} __PSGrid2DFloatDev;

typedef struct Kernel {
  cl_program program;

  void *extra;
  size_t extraSize;
  void *dtor;
  int noSource;
} Kernel;

typedef struct __attribute__((packed)) paramStruct {
  int inWidth;
  int outWidth;
  int inHeight;
  int outHeight;
  int inPixelPitch;
  int outPixelPitch;
  int inRowPitch;
  int outRowPitch;
  int offset;
  float delta;
} paramType;

typedef struct {

  float3 eye;

  float viewPlaneDistance;
  float zoom;

  float3 u, v, w;
} RT_Camera;

typedef struct {
  float4 m_localCenter;
  float4 m_extents;
  float4 mC;
  float4 mE;

  float m_radius;
  int m_faceOffset;
  int m_numFaces;
  int m_numVertices;

  int m_vertexOffset;
  int m_uniqueEdgesOffset;
  int m_numUniqueEdges;
  int m_unused;
} ConvexPolyhedronCL;

typedef struct {
  float _real;
  float _imag;
} clFloatComplex;

typedef struct {
  bool is_final;
  bool is_initialized;
  bool is_started;
  bool is_ready;
  bool reserved4;
  bool reserved5;
  bool reserved6;
  bool reserved7;
} WDFlags;

typedef struct xorwowStates {

  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  unsigned int v;

  unsigned int d;
} xorwowStates;

typedef struct {
  b2clDistanceProxy proxyA;
  b2clDistanceProxy proxyB;
  b2clSweep sweepA;
  b2clSweep sweepB;
  float tMax;
  float dummy;
} b2clTOIInput;

typedef struct apple_secure_notes_tmp {
  unsigned int ipad[8];
  unsigned int opad[8];

  unsigned int dgst[8];
  unsigned int out[8];

} apple_secure_notes_tmp_t;

typedef struct {
  w_vec_data_32 vec[16];
} data_32_wng;

typedef struct {
  data_32_wng lane[32];
} channel_vec_32_wng;

typedef struct {
  salt_t pbkdf2;
  uchar iv[16];
} lpcli_salt_t;

typedef struct __attribute__((aligned(1))) {
  uchar numEEDFs;
} LightPropertyInfo;

typedef struct {
  int n, lda, j0;
  short ipiv[32];
} dlaswp_params_t;

typedef struct {
  float3 box_start;
  float3 box_end;
  int child;
  int parent;
  int sibling;
  int data;
} KDTreeNodeHeader;

typedef struct intintLCGLinearOpenCompactCLHash_Bucket {
  int key;
  int value;
} intintLCGLinearOpenCompactCLHash_Bucket;

typedef struct {
  int intraE_contributors_const[3 * 64];
} kernelconstant_intracontrib;

typedef struct {
  int cells[4096];
} Mat64X64;

typedef struct {
  float4 shapeTransform[4];
  float4 linearVelocity;
  float4 angularVelocity;

  int softBodyIdentifier;
  int collisionShapeType;

  float radius;
  float halfHeight;
  int upAxis;

  float margin;
  float friction;

  int padding0;

} CollisionShapeDescription;

typedef struct office2013 {
  unsigned int encryptedVerifier[4];
  unsigned int encryptedVerifierHash[8];

} office2013_t;

typedef struct {
  union {
    Sph sph;
    Trgl trgl;
  };
  int matl_idx;
  enum { O_SPH, O_TRGL } type;
} Obj;

typedef struct {
  int quot, rem;
} div_t;

typedef struct {
  signed char decimation_mode;
  signed char quantization_mode;
  signed char is_dual_plane;
  signed char permit_encode;
  signed char permit_decode;
  float percentile;
} block_mode;

typedef struct {
  Spectrum radiance;
  float alpha;
} RandomSampleWithAlphaChannel;

typedef struct {
  float3 amod;
  float3 bs;
  float3 bis;
} processed_line3;

typedef struct {

  short genotipo[1 * 10];
  float aptidao;

} individuo;

typedef struct {
  float3 pos;
  float3 dir;
} TRay;

typedef struct {
  short coeff[16];
} block_t;

typedef struct {
  float16 Sum;
  float16 C;
} KahanAccumulator16;

typedef struct Context {
  const float4 mins;
  const float4 maxs;
  const matrix3 rot;
} Context;

typedef struct {

  __global float *pHbUp;
  __global float *pHb;
  __global float *pHbDn;
  __global float *pHbSqrtUp;
  __global float *pHbSqrt;
  __global float *pHbSqrtDn;
  __global float *pSpeedUp;
  __global float *pSpeed;
  __global float *pSpeedDn;
  CFluxVecPtr inPtrUp;
  CFluxVecPtr inPtr;
  CFluxVecPtr inPtrDn;

  CFluxVecPtr outPtr;

  CFluxVec flow;

  CFluxVec cur;
  float curHb;
  float curHbSqrt;
  float curSpeed;

} CFlux2DLocalPtrs;

typedef struct sha512_hmac_ctx {
  sha512_ctx_t ipad;
  sha512_ctx_t opad;

} sha512_hmac_ctx_t;

typedef struct streebog256_ctx_vector {
  unsigned long long h[8];

  unsigned long long s[8];

  unsigned long long n[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __constant unsigned long long (*s_sbob_sl64)[256];

} streebog256_ctx_vector_t;

typedef struct _ModelMatchParams {
  int idx;
  float dist;
} ModelMatchParams;

typedef struct b3GpuChildShape b3GpuChildShape_t;
struct b3GpuChildShape {
  float4 m_childPosition;
  float4 m_childOrientation;
  union {
    int m_shapeIndex;
    int m_capsuleAxis;
  };
  union {
    float m_radius;
    int m_numChildShapes;
  };
  union {
    float m_height;
    int m_collidableShapeIndex;
  };
  int m_shapeType;
};

typedef struct {
  uint32_t aaa;
  int mm, nn, rr, ww;
  uint32_t wmask, umask, lmask;
  int shift0, shift1, shiftB, shiftC;
  uint32_t maskB, maskC;
  int i;
  uint32_t *state;
} mt_struct;

typedef struct krb5tgs {
  unsigned int account_info[512];
  unsigned int checksum[4];
  unsigned int edata2[2560];
  unsigned int edata2_len;

} krb5tgs_t;

typedef struct {
  unsigned int length;
  unsigned int eapol[(256 + 64) / 4];
  unsigned int eapol_size;
  unsigned int data[(64 + 12) / 4];
  uchar salt[36];
} wpapsk_salt;

typedef struct heap_t {
  float v;
  int idx;
} heap_t;

typedef struct mode45_parameters {
  int qep[8];
  unsigned int qblock[2];
  int aqep[2];
  unsigned int aqblock[2];
  int rotation;
  int swap;
} mode45_parameters;

typedef struct s_paraboloid {
  float3 pos;
  float k;
  float m;
  float tex_scale;
} t_paraboloid;

typedef struct {
  int group_id;
  int padding[3];
} ShapeAdditionalData;

typedef struct _cl_pid_t {
  float mv;
  float sv;
  float pv;
  float e;

  float kp;
  float p_out;

  float ki;
  float iv;
  float em[(10)];
  int idx;
  float i_out;

  float kd;
  float dp;
  float d_out;
} cl_pid_t;

typedef struct {
  float4 o;
  float4 d;
  int2 extra;
  int2 padding;
} ray;

typedef struct hit {
  int object;
  float3 position;
  float3 normal;
  ray hitter;
  int hit;
  float distance;
} hit;

typedef struct subt {
  subt_rec recs[1024];
  ushort av_recs[1024 + 1];
} subt;

typedef struct {
  long vertexOffset;
  int vertexLength;
  int edgeLength;
} DataContext;

typedef struct {
  unsigned int v[(64 + 3) / 4];
} pbkdf2_hash;

typedef struct _AES_CMAC_CTX {
  AES_CTX aesctx;
  uint8_t X[16];
  uint8_t M_last[16];
  unsigned int M_n;
} AES_CMAC_CTX;

typedef struct s_torus {
  float3 pos;
  float big_r;
  float r;
} t_torus;

typedef struct {
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
} ReconstructionSample;

typedef struct {
  float rng0, rng1;
  unsigned int pixelIndex, pass;

  Spectrum radiance;
  float alpha;
} SobolSampleWithAlphaChannel;

typedef struct {
  unsigned int length;
  uchar v[256];
} pbkdf2_password;

typedef struct __attribute__((aligned(4))) Stage {
  int first __attribute__((aligned(4)));
  int ntrees __attribute__((aligned(4)));
  float threshold __attribute__((aligned(4)));
} Stage;

typedef struct {

  float4 p0;
  float4 p1;
  float4 p2;

  float4 n0;
  float4 n1;
  float4 n2;

  float4 size;

  int type;

  int index;

  int materialId;

  float2 vt0;
  float2 vt1;
  float2 vt2;
} Primitive;

typedef struct {
  unsigned int dataIndex;
} HitPointTriangleAOVTexParam;

typedef struct tacacs_plus {
  unsigned int session_buf[16];

  unsigned int ct_data_buf[64];
  unsigned int ct_data_len;

  unsigned int sequence_buf[16];

} tacacs_plus_t;

typedef struct {
  float stage_threshold;
  int num_weak_classifiers;
  int classifier_start_index;
} stage;

typedef struct {
  int start;
  int len;
  int vab;
} VabIndex;

typedef struct {
  float alpha;
} AlphaPixel;

typedef struct MatchEntry_t {
  int index;
  int value;
} MatchEntry;

typedef struct md4_hmac_ctx_vector {
  md4_ctx_vector_t ipad;
  md4_ctx_vector_t opad;

} md4_hmac_ctx_vector_t;

typedef struct NestedBool {
  int x;
  struct InnerNestedBool {
    bool boolField;
  } inner;
} NestedBool;

typedef struct blake2b_ctx {
  unsigned long long m[16];
  unsigned long long h[8];

  unsigned int len;

} blake2b_ctx_t;

typedef struct buffer_t {

  uint64_t dev;

  uint8_t *host;

  int extent[4];
  int stride[4];

  int min[4];

  int elem_size;

  bool host_dirty;

  bool dev_dirty;
} buffer_t;

typedef struct telegram {
  unsigned int data[72];

} telegram_t;

typedef struct {
  float r, g, b;
  int specularBounce;
} MirrorParam;

typedef struct {
  MatteParam matte;
  MirrorParam mirror;
  float matteFilter, totFilter, mattePdf, mirrorPdf;
} MatteMirrorParam;

typedef struct _PAPhysicsWindData {
  float strength, noise;
  PAPhysicsFalloffData falloff;
} PAPhysicsWindData;

typedef struct {
  float4 m_plane;
  int m_indexOffset;
  int m_numIndices;
} btGpuFace;

typedef struct {

  char *device;
  int output;
  char *root;
  int devices;
  int profile;
  int batch_header;
  int show_rules;
  char *rule;

  char *image;
  struct path_or_real *weight;
  struct path_or_real *xweight;
  char *mask;
  char *psf;
  double offset;
  struct path_or_real *gain;
  double bscale;

  int nlive;
  int ins;
  int mmodal;
  int ceff;
  double acc;
  double tol;
  double shf;
  int maxmodes;
  int feedback;
  int updint;
  int seed;
  int resume;
  int maxiter;

  int ds9;
  char *ds9_name;
} options;

typedef struct {
  float4 vertices[4];
  float4 normal;
  float4 float3;
} TAreaLight;

typedef struct {
  unsigned int tex1Index, tex2Index, tex3Index;
} MakeFloat3TexParam;

typedef struct {
  int que[100];
  int que_start;
  int que_end;
  int que_max;
} TQueI;

typedef struct tag_camera_params

{
  float16 viewMat;
  float3 bbox_bl;
  float3 bbox_tr;
  unsigned int ortho;
} camera_params;

typedef struct {
  float3 old_position, new_position, next_velocity;
} advection_result;

typedef struct {
  int start;
  int len;
} FaceIndex;

typedef struct LodePNGColorProfile {
  unsigned colored;
  unsigned key;
  unsigned short key_r;
  unsigned short key_g;
  unsigned short key_b;
  unsigned alpha;
  unsigned numcolors;
  unsigned char palette[1024];
  unsigned bits;
} LodePNGColorProfile;

typedef struct {
  float uScale, vScale, uDelta, vDelta;
} UVMappingParam;

typedef struct {
  unsigned int index;
  char type;
} TObject;

typedef struct _environment_t {
  float4 env_col_and_clamp;
  unsigned int env_map;
  float pad[3];
} environment_t;

typedef struct {
  bool is_QVECTOR;
  bool conv_start;
  bool conv_done[(3 + 5 - 1)];
  bool pe_output_relu;

  int filter_read_addr;
  int filter_write_addr;
  char filter_read_fw_vec;
  bool filter_bias_read_page;
} PeControlSignal;

typedef struct {
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
} cv;

typedef struct sha1_hmac_ctx {
  sha1_ctx_t ipad;
  sha1_ctx_t opad;

} sha1_hmac_ctx_t;

typedef struct {
  float t;
  float b1, b2;
  unsigned int index;
} RayHit;

typedef struct _PathVertex {
  float3 position;
  float3 shading_normal;
  float3 geometric_normal;
  float2 uv;
  float pdf_forward;
  float pdf_backward;
  float3 flow;
  float3 unused;
  int type;
  int material_index;
  int flags;
  int padding;
} PathVertex;

typedef struct __attribute__((aligned(4))) ScaleData {
  float scale __attribute__((aligned(4)));
  int szi_width __attribute__((aligned(4)));
  int szi_height __attribute__((aligned(4)));
  int layer_ofs __attribute__((aligned(4)));
  int ystep __attribute__((aligned(4)));
} ScaleData;

typedef struct ParticleRigidBodyParams {
  float smoothing_distance;
  float simulation_scale;

  float4 gravity;

  float friction_dynamic;
  float friction_static;
  float friction_static_threshold;
  float shear;
  float attraction;
  float spring;
  float dampening;

  float EPSILON;
  float velocity_limit;

  int num;
  int max_num;

} ParticleRigidBodyParams;

typedef struct decode_args_s {
  const char *in;
  char *out;
  const size_t strlength;
} decode_args_t;

typedef struct {
  float fx;
  float fy;
  float fz;
  int uw;
} btAABBCL;

typedef struct b3ContactConstraint4 b3ContactConstraint4_t;

struct b3ContactConstraint4 {
  float4 m_linear;
  float4 m_worldPos[4];
  float4 m_center;
  float m_jacCoeffInv[4];
  float m_b[4];
  float m_appliedRambdaDt[4];
  float m_fJacCoeffInv[2];
  float m_fAppliedRambdaDt[2];

  unsigned int m_bodyA;
  unsigned int m_bodyB;
  int m_batchIdx;
  unsigned int m_paddings;
};

typedef struct MyStruct {
  int anint;
  float afloat;
  int threeints[3];
} MyStruct;

typedef struct {
  float p;
  float cg;
  float c;
  float pp;
  float cgp;
  float cp;
  float dt;
  float h;
  float k;
  float m;
  float n;
  float fitness;
} member;

typedef struct empty_struct {
} empty_struct;

typedef struct num_regs_nested_struct {
  int x;
  struct nested {
    char x;
    long y;
  } inner;
} num_regs_nested_struct;

typedef struct {
  unsigned long long number_of_photons;
  unsigned int number_of_photons_per_voxel;
  unsigned int n_layers;
  unsigned int n_bulks;
  unsigned int start_weight;

  char outp_filename[500];
  char inp_filename[500];

  long begin, end;
  char AorB;

  DetStruct det;
  LayerStruct *layers;
  BulkStruct *bulks;
  IncStruct inclusion;

  int grid_size;

  float esp;

  float xi;
  float yi;
  float zi;
  float dir;

  float n_down;

  int fhd_activated;
  int do_fl_sim;
  int bulk_method;

  short *bulk_info;

  char bulkinfo_filename[500];

  int do_temp_sim;

} SimulationStruct;

typedef struct def_SolidObject {
  unsigned int type;

  union {
    Plane plane;
    Sphere sphere;
    Box box;
  } data;
} SolidObject;

typedef struct position_ {
  float4 b;
} position_t;

typedef struct _singquadg singquadg;

struct _singquadl {
  unsigned int nq;

  local const float *xqs;
  local const float *yqs;
  local const float *wqs;

  float4 bases;
};

typedef struct {
  int primitiveId;
  int materialId;
  float4 location;
  float4 float3;
} LightInformation;

struct PluginCodec_H323CapabilityExtension {
  unsigned int index;
  void *data;
  unsigned dataLength;
};

typedef struct _histogram {
  int ***grid;
  int x_dim, y_dim, z_dim;
  double origin[3];
  double delta[3][3];
  int count[3];
  int n_data_points;
  int norm_total;
} histogram_t;

typedef struct {
  float2 uv;
  float dist;
  bool hit;
} RTResult;

typedef struct {
  unsigned int mblim;
  unsigned int blim;
  unsigned int lim;
  unsigned int hev_thr;
} loop_filter_info;

typedef struct {
  float cellWidth;
  float invCellWidth;
  int cellMask;
  float halfCellWidth;
  float invHalfCellWidth;
} GridParams;

typedef struct {
  unsigned int total;
  unsigned int state[5];
  uchar buffer[64];
} SHA_CTX;

typedef struct {
  unsigned int x;
  unsigned int y;
} seed_value_t;

typedef struct _NhlBoundingBox {
  int set;
  float t;
  float b;
  float l;
  float r;
} NhlBoundingBox;

typedef struct krb5asrep {
  unsigned int account_info[512];
  unsigned int checksum[4];
  unsigned int edata2[5120];
  unsigned int edata2_len;

} krb5asrep_t;

typedef struct {

  Spectrum lightRadiance;

  float lastPdfW;
  int lastSpecular;
} PathStateDirectLight;

typedef struct {
  float4 R;
  float4 G;
  float4 B;
} pColor;

typedef struct blake2s_state_t {
  unsigned int h[8];
  unsigned int t[2];
  unsigned int f[2];
  uchar buf[2 * 64U];
  unsigned int buflen;
} blake2s_state;

typedef struct intintIdentityPerfectCLHash_TableData {
  int hashID;
  unsigned int numBuckets;
  char compressFuncData;
} intintIdentityPerfectCLHash_TableData;

typedef struct {

  size_t size;

  size_t lower_bound;

  size_t accessed_length;
} nanos_region_dimension_internal_t;

typedef struct {
  uint8_t length;
  char v[255 + 1];
} sha512_key;

typedef struct int_single {
  int a;
} int_single;

typedef struct b3Contact4Data b3Contact4Data_t;

struct b3Contact4Data {
  float4 m_worldPosB[4];

  float4 m_worldNormalOnB;
  unsigned short m_restituitionCoeffCmp;
  unsigned short m_frictionCoeffCmp;
  int m_batchIdx;
  int m_bodyAPtrAndSignBit;
  int m_bodyBPtrAndSignBit;

  int m_childIndexA;
  int m_childIndexB;
  int m_unused1;
  int m_unused2;
};

typedef struct infonode {
  ulong id;
  float x;
  float y;
} infonode;

typedef struct {

  float fps, erp;

  union {
    __global float4 *m_J1linearAxisFloat4;
    __global float *m_J1linearAxis;
  };
  union {
    __global float4 *m_J1angularAxisFloat4;
    __global float *m_J1angularAxis;
  };
  union {
    __global float4 *m_J2linearAxisFloat4;
    __global float *m_J2linearAxis;
  };
  union {
    __global float4 *m_J2angularAxisFloat4;
    __global float *m_J2angularAxis;
  };

  int rowskip;

  __global float *m_constraintError;
  __global float *cfm;

  __global float *m_lowerLimit;
  __global float *m_upperLimit;

  __global int *findex;

  int m_numIterations;

  float m_damping;
} b3GpuConstraintInfo2;

typedef struct __blake2sp_state {
  blake2s_state S[8][1];
  blake2s_state R[1];
  uint8_t buf[8 * 64];
  size_t buflen;
} blake2sp_state;

typedef struct pbkdf2_sha1_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[32];
  unsigned int out[32];

} pbkdf2_sha1_tmp_t;

typedef struct LightSample {
  float3 P;
  float3 Ng;
  float3 D;
  float t;
  float pdf;
  float eval_fac;
  int object;
  int prim;
  int shader;
  int lamp;
} LightSample;

typedef struct cuComplex {
  float x;
  float y;
} cuComplex;

typedef struct {
  int error_block;
  int block_mode;
  int partition_count;
  int partition_index;
  int color_formats[4];
  int color_formats_matched;
  int color_values[4][12];
  int color_quantization_level;
  uint8_t plane1_weights[64];
  uint8_t plane2_weights[64];
  int plane2_color_component;
  int constant_color[4];
} symbolic_compressed_block;

typedef struct {
  unsigned int texIndex;
} FresnelApproxNTexParam;

typedef struct {
  int n, lda, j0, npivots;
  short ipiv[32];
} slaswp_params_t2;

typedef struct ONB {
  float4 u;
  float4 v;
  float4 w;
} ONB;

typedef struct {
  float scale;
  int stride;
  int kernelSize;
  int pad;
  int dilation;
  int inputChannel;
  int inputHeight;
  int inputWidth;
  int outputChannel;
  int outputHeight;
  int outputWidth;
  int inputTotalDataNum;
  int outputTotalDataNum;
} NetParam;

typedef struct {
  loop_filter_thresh_ocl lfthr[63 + 1];
  uint8_t lvl[8][4][2];
  uint8_t mode_lf_lut[14];
} loop_filter_info_n_ocl;

typedef struct atmi_task_s {

  atmi_state_t state;

  atmi_tprofile_t profile;
} atmi_task_t;

typedef struct halide_buffer_t {

  uint64_t device;

  const struct halide_device_interface_t *device_interface;

  uint8_t *host;

  uint64_t flags;

  int dimensions;

  halide_dimension_t *dim;

  void *padding;
} halide_buffer_t;

typedef struct {
  Matrix3x3 m_basis;
  float4 m_origin;
} b3Transform;

typedef struct clb2Velocity {
  float vx;
  float vy;
  float w;
} clb2Velocity;

typedef struct {
  unsigned int matBaseIndex;
  unsigned int ksTexIndex;
  unsigned int nuTexIndex;
  unsigned int nvTexIndex;
  unsigned int kaTexIndex;
  unsigned int depthTexIndex;
  unsigned int indexTexIndex;
  int multibounce;
} GlossyCoatingParam;

typedef struct WebCLValidator *clv_program;

clv_program clvValidate(const char *input_source,
                        const char **active_extensions,
                        const char **user_defines, void *pfn_notify,
                        void *notify_data, cl_int *errcode_ret);

typedef enum {

  CLV_PROGRAM_VALIDATING,

  CLV_PROGRAM_ILLEGAL,

  CLV_PROGRAM_ACCEPTED_WITH_WARNINGS,

  CLV_PROGRAM_ACCEPTED
} clv_program_status;

typedef struct sha384_hmac_ctx_vector {
  sha384_ctx_vector_t ipad;
  sha384_ctx_vector_t opad;

} sha384_hmac_ctx_vector_t;

typedef struct {
  uint4 s0, s1, s2, s3;
} v4uint4;

typedef struct {

  unsigned short int m_quantizedAabbMin[3];
  unsigned short int m_quantizedAabbMax[3];

  int m_rootNodeIndex;

  int m_subtreeSize;
  int m_padding[3];
} b3BvhSubtreeInfo;

typedef struct _map {
  int key;
  float value;
} mapx;

typedef struct wpapmk_tmp {
  unsigned int out[8];

} wpapmk_tmp_t;

typedef struct JobDescription {
  int JobID;
  int JobType;
  int numThreads;
  int params;
} JobDescription;

typedef struct _spherical_harmonic {
  double legendre;
  double real;
  double imaginary;

} spherical_harmonic_t;

typedef struct {
  IntrinsicParameter intrinsic;
  ExtrinsicParameter extrinsic;
  float radius;
  float distort_coeff[4];
  float c_coeff[4];
} FisheyeInfo;

typedef struct {
  PathState state;

  Spectrum throughput;
  BSDF bsdf;

  Seed seedPassThroughEvent;

  int albedoToDo, photonGICacheEnabledOnLastHit, photonGICausticCacheUsed,
      photonGIShowIndirectPathMixUsed,

      throughShadowTransparency;
} GPUTaskState;

typedef struct {
  int current_cell;
  char next_subcell;
  int children[8];
} stack_entry;

typedef struct s_triangle {
  float3 d1;
  float3 d2;
  float3 d3;
  float tex_scale;
} t_triangle;

typedef struct {
  float x, y, z;
} Vec;

typedef struct {
  int numPoints;
  int countOn;
  int countAbove;
  int countBelow;
  float euclideanThreshold;
  Vec n0;
  float dist;
  int tolerance;
  int acc;
  int nacc;

} Result;

typedef struct zip2 {
  unsigned int type;
  unsigned int mode;
  unsigned int magic;
  unsigned int salt_len;
  unsigned int salt_buf[4];
  unsigned int verify_bytes;
  unsigned int compress_length;
  unsigned int data_len;
  unsigned int data_buf[2048];
  unsigned int auth_len;
  unsigned int auth_buf[4];

} zip2_t;

typedef struct {
  float v, x, y, z;

} FOUR_VECTOR;

struct pkzip {
  unsigned char hash_count;
  unsigned char checksum_size;
  unsigned char version;

} __attribute__((packed));

typedef struct _param_global {
  double alpha;
  double last_alpha;
  param_t *type_params;
} param_g;

typedef struct {
  unsigned int array[2];
} boundary_index_array_2;

typedef struct {
  unsigned int tex1Index, tex2Index;
} LessThanTexParam;

typedef struct {
  DotFilterVector filter_data;
  BiasBnParam bias_bn_data;
  int n_inc;
  bool data_valid;
} PeInputFilter;

typedef struct LodePNGDecoderSettings {
  LodePNGDecompressSettings zlibsettings;

  unsigned ignore_crc;

  unsigned color_convert;

  unsigned read_text_chunks;

  unsigned remember_unknown_chunks;

} LodePNGDecoderSettings;

typedef struct {
  float16 viewMatrix;
  float16 viewMatrixInv;
  float16 projMatrix;
  float16 projMatrixInv;
  float16 vpMatrixInv;

  float16 motionBlurMatrix;

  float3 position;
  float3 lookVector;

} cl_camera;

typedef struct LodePNGColorMode {

  unsigned bitdepth;
  unsigned char *palette;
  size_t palettesize;
  unsigned key_defined;
  unsigned key_r;
  unsigned key_g;
  unsigned key_b;
} LodePNGColorMode;

typedef struct LodePNGTime {
  unsigned year;
  unsigned month;
  unsigned day;
  unsigned hour;
  unsigned minute;
  unsigned second;
} LodePNGTime;

typedef struct LodePNGInfo {

  unsigned compression_method;
  unsigned filter_method;
  unsigned interlace_method;
  LodePNGColorMode float3;
  unsigned background_defined;
  unsigned background_r;
  unsigned background_g;
  unsigned background_b;
  size_t text_num;
  char **text_keys;
  char **text_strings;

  size_t itext_num;
  char **itext_keys;
  char **itext_langtags;
  char **itext_transkeys;
  char **itext_strings;

  unsigned time_defined;
  LodePNGTime time;

  unsigned phys_defined;
  unsigned phys_x;
  unsigned phys_y;
  unsigned phys_unit;
  unsigned char *unknown_chunks_data[3];
  size_t unknown_chunks_size[3];

} LodePNGInfo;

typedef struct LodePNGState {

  LodePNGDecoderSettings decoder;

  LodePNGDecoderSettings encoder;

  LodePNGColorMode info_raw;
  LodePNGInfo info_png;
  unsigned error;

} LodePNGState;

typedef struct {
  buffer_32 alt_result[8];
  buffer_32 temp_result[(16 / 4)];
  buffer_32 p_sequence[(24 / 4)];
} sha256_buffers;

typedef struct {
  int *int_pointer;
  char *char_pointer;
  float *float_pointer;
  float4 *vector_pointer;
} PrivatePrimitivePointerStruct;

typedef struct {
  unsigned int baseColorTexIndex;
  unsigned int subsurfaceTexIndex;
  unsigned int roughnessTexIndex;
  unsigned int metallicTexIndex;
  unsigned int specularTexIndex;
  unsigned int specularTintTexIndex;
  unsigned int clearcoatTexIndex;
  unsigned int clearcoatGlossTexIndex;
  unsigned int anisotropicTexIndex;
  unsigned int sheenTexIndex;
  unsigned int sheenTintTexIndex;
  unsigned int filmAmountTexIndex;
  unsigned int filmThicknessTexIndex;
  unsigned int filmIorTexIndex;
} DisneyParam;

typedef struct {
  float3 eyePos;
  float3 targetPos;
  float3 up;
  float3 voxelBounds;
  float3 voxelBounds2;
  float3 voxelBoundsMin;
  float3 voxelBoundsMax;
  float3 invVoxelScale;
  float3 skyColor1;
  float3 skyColor2;
  int4 voxelRes;
  int2 resolution;
  float invAspect;
  float time;
  float fov;
  int maxIter;
  int maxVoxelIter;
  float maxDist;
  float startDist;
  float eps;
  int aoIter;
  float aoStepDist;
  float aoAmp;
  float voxelSize;
  float groundY;
  int shadowIter;
  int reflectIter;
  float shadowBias;
  float lightScatter;
  float minLightAtt;
  float gamma;
  float exposure;
  float dof;
  float frameBlend;
  float fogPow;
  float flareAmp;
  int mcTableLength;
  uchar isoVal;
  uchar numLights;
  float4 lightPos[4];
  float4 lightColor[4];
  TMaterial materials[4];
} TRenderOpts;

typedef struct {
  float3 o;
  float3 d;
} RT_Ray;

typedef struct {
  int m_a;
  int m_b;
  unsigned int m_idx;
} Elem;

typedef struct {
  float3 c;
  float3 n;
  float3 s;
  float3 e;
  float3 w;
  float3 u;
  float3 d;
} Neighbors_float3;

typedef struct {
  unsigned int textureIndex;
  unsigned int incrementIndex;
} RoundingTexParam;

typedef struct {

  Transform rasterToCamera;
  Transform cameraToWorld;

  float yon, hither;
  float shutterOpen, shutterClose;

  MotionSystem motionSystem;
  InterpolatedTransform interpolatedTransforms[8];
} CameraBase;

typedef struct QuadNode {
  float4 bbox[2 * 3];
  unsigned int child_nodes[4];
  unsigned int number_of_spheres[4];
} QuadNode;

typedef struct {
  ushort data[16];
} lane_sign;

typedef struct {
  float4 parameters[2];
  int nbPrimitives;
  int startIndex;
  int2 indexForNextBox;
} BoundingBox;

typedef struct {
  int keywordOffset;
  int keywordSize;
} KEY_T;

typedef struct {
  uint8_t length;
  char v[255 + 1];
} xsha512_key;

typedef struct {
  unsigned int stream_count_kernel__histogram_matrix[1024];
  unsigned int prefix_scan_kernel__unused_vector_level_prefix_sums[256];
} WclLocals;

typedef struct ge25519_pniels_t {
  unsigned int ysubx, xaddy, z, t2d;
} ge25519_pniels;

typedef struct {
  int start_index;
  int end_index;
  float start_continuous_index;
  float end_continuous_index;
} GPUImageFunction1D;

typedef struct bs_word {
  unsigned int b[32];

} bs_word_t;

typedef struct s_light {
  float4 position;
  float4 float3;
} t_light;

typedef struct pbkdf2_sha512 {
  unsigned int salt_buf[32];

} pbkdf2_sha512_t;

typedef struct {
  __local WclLocals *wcl_locals_min;
  __local WclLocals *wcl_locals_max;
} WclLocalLimits;

typedef struct {
  WclConstantLimits cl;
  WclGlobalLimits gl;
  WclLocalLimits ll;
  WclPrivates pa;
} WclProgramAllocations;

typedef struct struct_4regs {
  int x;
  int y;
  int z;
  int w;
} struct_4regs;

typedef struct TfLiteQuantizationParams {
  float scale;
  int zero_point;
} TfLiteQuantizationParams;

typedef struct s_kparam {

  int sampling_frequency;

  int N;

  int tracksize;

  float minfreq;
  float maxfreq;
  int start_sample;
  int stop_sample;
} t_kparam;

typedef struct {
  unsigned int N_;
  unsigned int SIZE_;
  unsigned int STRIPES_;
  unsigned int WIDTH_;
  unsigned int PCOUNT_;
  unsigned int TARGET_;
  unsigned int LIMIT13_;
  unsigned int LIMIT14_;
  unsigned int LIMIT15_;
  unsigned int GCN_;
} config_t;

typedef struct {

  float4 m_relpos1CrossNormal;
  float4 m_contactNormal;

  float4 m_relpos2CrossNormal;

  float4 m_angularComponentA;
  float4 m_angularComponentB;

  float m_appliedPushImpulse;
  float m_appliedImpulse;
  int m_padding1;
  int m_padding2;
  float m_friction;
  float m_jacDiagABInv;
  float m_rhs;
  float m_cfm;

  float m_lowerLimit;
  float m_upperLimit;
  float m_rhsPenetration;
  int m_originalConstraint;

  int m_overrideNumSolverIterations;
  int m_frictionIndex;
  int m_solverBodyIdA;
  int m_solverBodyIdB;

} b3SolverConstraint;

typedef struct {
  unsigned char S[256];

  unsigned int wtf_its_faster;

} RC4_KEY;

typedef struct {
  int boundary_type;
  unsigned int boundary_index;
} condensed_node;

typedef struct {
  unsigned int v[4];
} password_hash_t;

typedef struct MQDecoder {
  MQEncoder encoder;
  unsigned char NT;
  int Lmax;
} MQDecoder;

typedef struct UserData {
  int x;
  int y;
  int z;
  int w;
} UserData;

typedef struct {
  int m_bodyAPtrAndSignBit;
  int m_bodyBPtrAndSignBit;
  int m_originalConstraintIndex;
  int m_batchId;
} b3BatchConstraint;

typedef struct {
  float4 bboxes[2][3];
  int4 children;
} QBVHNode;

typedef struct {

  union {
    unsigned char c[8][8][sizeof(int)];
    int v[8][8];
  } xkeys;

  int keys_changed;
} DES_bs_transfer;

typedef struct {
  float3 direction;
  float coefficient;
} Speaker;

typedef struct {
  unsigned int layers;
  unsigned int height;
  unsigned int width;
  unsigned int count;
} SurfInfo;

typedef struct {
  unsigned int ukey[8];

  unsigned int hook_success;

} seven_zip_hook_t;

typedef struct {
  float _0;
  int _1;
  int _2;
} Tuple_float_int_int;

typedef struct {
  unsigned int total;
  ulong state[8];
  uchar buffer[128];
} SHA512_CTX;

typedef struct __attribute__((aligned(64))) GpuHidHaarClassifierCascade {
  int count __attribute__((aligned(4)));
  int is_stump_based __attribute__((aligned(4)));
  int has_tilted_features __attribute__((aligned(4)));
  int is_tree __attribute__((aligned(4)));
  int pq0 __attribute__((aligned(4)));
  int pq1 __attribute__((aligned(4)));
  int pq2 __attribute__((aligned(4)));
  int pq3 __attribute__((aligned(4)));
  int p0 __attribute__((aligned(4)));
  int p1 __attribute__((aligned(4)));
  int p2 __attribute__((aligned(4)));
  int p3 __attribute__((aligned(4)));
  float inv_window_area __attribute__((aligned(4)));
} GpuHidHaarClassifierCascade;

typedef struct office2007 {
  unsigned int encryptedVerifier[4];
  unsigned int encryptedVerifierHash[5];

  unsigned int keySize;

} office2007_t;

typedef struct {
  float weight;

  long assign;
  float cost;
} Point_Struct;

typedef struct sip {
  unsigned int salt_buf[32];
  unsigned int salt_len;

  unsigned int esalt_buf[48];
  unsigned int esalt_len;

} sip_t;

typedef struct {
  unsigned int kdTexIndex;
  unsigned int sigmaTexIndex;
} RoughMatteParam;

typedef struct streebog256_hmac_ctx_vector {
  streebog256_ctx_vector_t ipad;
  streebog256_ctx_vector_t opad;

} streebog256_hmac_ctx_vector_t;

typedef struct {
  float4 error_weights[216];
} error_weight_block_orig;

typedef struct struct_arr33 {
  int arr[33];
} struct_arr33;

typedef struct {
  uint64_t v[4];
} uint256_t;

typedef struct {
  uint32_t version[1024];
  uint32_t prevBlockHash[8 * 1024];
  uint32_t merkleRoot[8 * 1024];
  uint32_t timestamp[1024];
  uint32_t nBits[1024];
  uint32_t nonce[1024];

  uint256_t blockHeaderHash;

} primecoinBlockcl_t;

typedef struct {
  unsigned int cmds[32];

} kernel_rule_t;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_Le __attribute__((aligned(4)));
  float multiplier;
  unsigned int idx_Dist2D;
} ImageBasedEnvLElem;

typedef struct {
  float3 point;
  float3 viewVector;
  float3 viewVectorNotRotated;
  float3 normal;
  float3 lightVect;
  float distThresh;
  float lastDist;
  float delta;
  float depth;
  int stepCount;
  int randomSeed;
  int objectId;
  bool invertMode;
  __global float4 *palette;
  int paletteSize;

} sShaderInputDataCl;

typedef struct {
  DotFeatureVector input_data;
  bool input_data_valid;
} PeInputData;

typedef struct sha1_double_salt {
  unsigned int salt1_buf[64];
  int salt1_len;

  unsigned int salt2_buf[64];
  int salt2_len;

} sha1_double_salt_t;

typedef struct _vertex_t {
  float p[3], n[3], b[3], t[2][2];
} vertex_t;

typedef struct {
  long s10, s11, s12, s20, s21, s22;
} mrg63k3a_state;

typedef struct {
  unsigned int sigmaATexIndex;
  unsigned int sigmaSTexIndex;
  unsigned int gTexIndex;
  int multiScattering;
} HomogenousVolumeParam;

typedef struct struct_arg {
  int i1;
  float f;
  int i2;
} struct_arg_t;

typedef struct __attribute__((aligned(16))) {
  uint32_t s[4];
} dag_t;

typedef struct {
  unsigned int type;
  float density;
  float friction;
  float hardness;
  float rigidity;
  float strength;
  float roughness;
  float elasticity;
} Substance;

typedef struct {
  Spectrum float3;
  float alpha;
} ConstFloat4Param;

typedef struct {
  unsigned int x;
  unsigned int c;
} rnd64x_state_t;

typedef struct {

  float4 p;
  Normal n;
  int isVolume;

  unsigned int lightsDistributionOffset;
  int pad[2];
} DLSCacheEntry;

typedef struct {
  ulong f1;
  ushort f2;
  ushort pad[3];
} ConstantFP80Ty;

typedef struct phpass_tmp {
  unsigned int digest_buf[4];

} phpass_tmp_t;

typedef struct clb2Transform {
  clb2Rotation rotation;
  float2 translation;
} clb2Transform;

typedef struct TestDesc {
  cl_uint widthA;
  cl_uint heightA;
  cl_uint widthB;
  cl_uint heightB;
  cl_uint srowA;
  cl_uint scolA;
  cl_uint srowB;
  cl_uint scolB;
  SubproblemDim dim;
  PGranularity pgran;
  bool transpose;
  bool packedImages;
} TestDesc;

typedef struct {
  uint32_t v[8];
} sha256_hash;

typedef struct PointData {

  float4 density;
  float4 float3;
  float4 color_normal;
  float4 color_lapl;
  float4 force;
  float4 surf_tens;
  float4 xsph;
  float4 viscosity;

} PointData;

typedef struct {
  float4 float3;
  float param;
  float firsthit;
  float channelIntensities[4];
} RayInfo;

typedef struct __attribute__((aligned(64))) {
  unsigned int width, height;
  float16 localToWorld;
} CameraHead;

typedef struct __attribute__((aligned(64))) {
  CameraHead head;
  uchar id;
  float virtualPlaneArea __attribute__((aligned(4)));
  float lensRadius;
  float objPDistance;
  float16 rasterToCamera __attribute__((aligned(64)));
} PerspectiveInfo;

typedef struct _PAPhysicsParticle {
  int lifetime;
  float x, y, z;
} PAPhysicsParticle;

typedef struct {
  float p1x;
  float p1y;
  float p2x;
  float p2y;

} ClLine;

typedef struct {
  unsigned int iterations;
  unsigned int outlen;
  unsigned int skip_bytes;
  uchar length;
  uchar salt[256];
  uchar iv[16];
  uchar aes_ct[16];
} agile_salt;

typedef struct {
  int complete;
  void *resultAddr;
} mallocWaitObj;

typedef struct large_struct_padding {
  char e0;
  int e1;
  char e2;
  int e3;
  char e4;
  char e5;
  short e6;
  short e7;
  char e8[3];
  long e9;
  int e10;
  char e11;
  int e12;
  short e13;
  char e14;
} large_struct_padding;

typedef struct __attribute__((aligned(16))) {
  float3 localOrigin;
  const global PerspectiveInfo *info;
} PerspectiveIDF;

typedef struct {
  uint32_t left_y[4];
  uint32_t above_y[4];
  uint32_t int_4x4_y;
  uint16_t left_uv[4];
  uint16_t above_uv[4];
  uint16_t int_4x4_uv;
  uint8_t lfl_y[64];
  uint8_t lfl_uv[16];
} LOOP_FILTER_MASK_OCL;

typedef struct {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
} decimal8;

typedef struct {
  float width;
  unsigned int borderTexIndex, insideTexIndex;
} WireFrameTexParam;

typedef struct {
  uint32_t uint32s[16];
} shuffle_t;

typedef struct {
  uint32_t objSize;
  clqueue_32 queue;
  uint32_t heap;
  uint32_t reduce_mem;
} clArrayList_32;

typedef struct _spiderinfo {
  int numberColony;
  int numberAgentPerColony;
  float backprobability;
  float pAttractSelf;
  float pAttractOther;
  float pSaturation;
  float constantWeight;
} spiderinfo;

typedef struct {
  float x_sum;
  float y_sum;
  int num_points;
} Accum;

typedef struct {
  double3 lo, hi;
} v2double3;

typedef struct keepass {
  unsigned int version;
  unsigned int algorithm;

  unsigned int keyfile_len;
  unsigned int keyfile[8];

  unsigned int final_random_seed[8];
  unsigned int transf_random_seed[8];
  unsigned int enc_iv[4];
  unsigned int contents_hash[8];

  unsigned int contents_len;
  unsigned int contents[75000];

  unsigned int expected_bytes[8];

} keepass_t;

typedef struct SimpleFog {
  float4 float3;
  int enabled;
  float distance;
  float steepness;
  int heightDispersion;
  float height;
  float heightSteepness;
} SimpleFog;

typedef struct {
  float3 collision_point, surface_normal;
  float penetration_depth;
  int collision_happened;
} collision;

typedef struct atmi_machine_s {

  unsigned int device_count_by_type[ATMI_DEVTYPE_ALL];

  atmi_device_t *devices_by_type[ATMI_DEVTYPE_ALL];
} atmi_machine_t;

typedef struct {
  uint4 x;
  uint4 c;
} mwc64xvec4_state_t;

typedef struct {
  lane_sign lane[32];
} channel_sign_vec;

typedef struct _Element {
  global float *internal;
  global float *external;
  float value;
} Element;

typedef struct queue_item_struct {
  int id;
  int value;
} queue_item_struct;

typedef struct {
  float3 min;
  float3 max;

  union {
    unsigned int leftChildIndex;
    unsigned int firstTriangleIndex;
  };
  unsigned int triangleCount;
} SubBvhNode;

typedef struct {
  unsigned int data[8];
} debug;

typedef struct _svm_model {
  svm_parameter *param;
  int nr_class;
  int svsLength;
  int svsWidth;

  __global const float *SV;

  __global const float *sv_coef;

  __constant float *rho;
  __constant int *label;

  __constant int *nSV;

  int free_sv;

} svm_model;

typedef struct {
  unsigned int iv_buf[4];
  unsigned int iv_len;

  unsigned int salt_buf[4];
  unsigned int salt_len;

  unsigned int crc;

  unsigned int data_buf[96];
  unsigned int data_len;

  unsigned int unpack_size;

} seven_zip_t;

typedef struct Pair {
  float first;
  float second;
} Pair;

typedef struct {
  float strength;
  unsigned int texIndex, offsetTexIndex;
} DistortTexParam;

typedef struct {
  unsigned int index;
  unsigned int hashid;
  uchar origin;
  uchar chainpos;
  uchar type;
  uchar reserved;
} fermat_t;

typedef struct {
  unsigned int cmds[15];

} gpu_rule_t;

typedef struct {
  uint32_t mults[(128 * 4 * 1024 / 16) * 3];
  uint32_t hash[(128 * 4 * 1024 / 16) * 3];
  int type[(128 * 4 * 1024 / 16) * 3];
  uint32_t prime[(128 * 4 * 1024 / 16) * 3];
} bresult_t;

typedef struct __attribute__((aligned(4))) {
  uchar texType;
  float value __attribute__((aligned(4)));
} FloatConstantTexture;

typedef struct VoxelOctree {

  float3 minPos;

  float3 maxPos;

  OctreeNode root;
} VoxelOctree;

typedef struct {
  unsigned int l[2];
  float w[512];
  float x[512 * 4];
} lmb1;

typedef struct {
  uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef struct __attribute__((aligned(4))) {
  ProceduralTextureHead head;
  float width __attribute__((aligned(4)));
  uchar reverse;
} Float3CheckerBoardBumpTexture;

typedef struct mogParams {
  float varThreshold;
  float backgroundRatio;
  float w0;
  float var0;
  float minVar;
} mogParams_t;

typedef struct {
  float4 m_row[4];
} Matrix4x4f;

typedef struct s_camera {
  float4 pos;
  float4 dir;
  float4 up;
  float4 right;
  float4 vpul;
  float2 vp_size;
} t_camera;

typedef struct {

  int r, g, b, a;
  int rgba;

} RT_Color;

typedef struct {
  Transform trans;
  int transSwapsHandedness;
} TriangleInstanceMeshParam;

typedef struct _Sampler {
  unsigned int index;
  unsigned int dimension;
  unsigned int scramble;
  unsigned int padding;
} Sampler;

typedef struct {
  int sign;
  int bits;
  int size;
  int extra_bits;
  int extra_size;
  __constant uint8_t *pcat;
} token;

typedef struct {
  int m_solveFriction;
  int m_maxBatch;
  int m_batchIdx;
  int m_nSplit;

} ConstBufferBatchSolve;

typedef struct {
  unsigned int keymic[16 / 4];
} mic_t;

typedef struct {
  int n;
} nanos_repeat_n_info_t;

typedef struct {
  uchar mask[(1024 / (8 * sizeof(uchar)))];
} zero_pad_mask_t;

typedef struct ripemd160_hmac_ctx {
  ripemd160_ctx_t ipad;
  ripemd160_ctx_t opad;

} ripemd160_hmac_ctx_t;

typedef struct s_material {
  float4 float3;
  float diffuse;
  float specular;
  float reflection;
  t_texture texture;
} t_material;

typedef struct AnsTableEntry_Struct {
  ushort freq;
  ushort cum_freq;
  uchar symbol;
} AnsTableEntry;

typedef struct single_struct_element_struct_arg {
  struct inner s;
} single_struct_element_struct_arg_t;

typedef struct {
  float m_hitFraction;
  int m_hitResult0;
  int m_hitResult1;
  int m_hitResult2;
  float4 m_hitPoint;
  float4 m_hitNormal;
} b3RayHit;

typedef struct _spider {
  int posision, lastposision;
  unsigned char colony;
} spider;

typedef struct {
  union {
    float4 minExtent;

    int4 leftChild;

    int4 meshInstance;

    int4 firstTriIndex;
  };

  union {
    float4 maxExtent;

    int4 rightChild;

    int4 numTriangles;
  };
} BvhNode;

typedef struct b3InertiaData b3InertiaData_t;

struct b3InertiaData {
  b3Mat3x3 m_invInertiaWorld;
  b3Mat3x3 m_initInvInertia;
};

typedef struct {
  __global float2 *data;
  __global float4 *loc;
} sampleArrayStruct;

typedef struct {
  volatile unsigned int head;
  volatile unsigned int tail;
} clqueue;

typedef struct _geom geom;

struct _geom {
  unsigned int dim;

  unsigned int n;

  global const float *v;
  global const unsigned int *p;
  global const float *g;

  local float3 (*vl)[3];
  local uint3 *pl;
  local float *gl;
};

typedef struct oraclet_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[16];
  unsigned long long out[16];

} oraclet_tmp_t;

typedef struct {
  su3vec e0;
  su3vec e1;
  su3vec e2;
  su3vec e3;
} spinor __attribute__((aligned(32)));

typedef struct {
  Inner ll;
} Outer;

struct mat4 {
  float4 x, y, z, w;
};

typedef struct {

  unsigned int istate[5];
  unsigned int ostate[5];
  unsigned int buf[5];
  unsigned int out[4];

} temp_buf;

typedef struct {
  union {
    float data[4];
    struct {
      float x;
      float y;
      float z;
      float w;
    };
  };
} PointXYZ;

typedef struct _PAEmitter {
  PAPhysicsParticle float4;

  float birthRate, birthRateNoise;
  float lifetime, lifetimeNoise;
  float initialVelocity, initialVelocityNoise;
  float emissionAngleStart, emissionAngleEnd, emissionAngleNoise;
} PAEmitter;

typedef struct {
  unsigned int matIndex;
  union {
    unsigned int opsCount;
  } opData;
} MaterialEvalOp;

typedef struct {
  unsigned int t00Index, t01Index, t10Index, t11Index;
} BilerpTexParam;

typedef struct krb5pa_18_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];
  unsigned int dgst[16];
  unsigned int out[16];

} krb5pa_18_tmp_t;

typedef struct {
  unsigned int l;
  float r;
  float x[512 * 4];
} lmb2;

typedef struct {

  __global HlbvhNode const *nodes;

  __global bbox const *bounds;

  __global float3 const *vertices;

  __global int const *faces;

  __global ShapeData const *shapes;

  __global int const *extra;
} SceneData;

typedef struct {
  unsigned int m_src_width;
  unsigned int m_src_height;
  unsigned int m_xdim;
  unsigned int m_ydim;
  unsigned int m_zdim;

  int quantization_mode_table[17][128];

  float m_target_bitrate;
  int batch_size;

  int m_rgb_force_use_of_hdr;
  int m_alpha_force_use_of_hdr;
  int m_perform_srgb_transform;
  int m_ptindex;
  int m_texels_per_block;
  unsigned int m_width_in_blocks;
  unsigned int m_height_in_blocks;

  float sin_table[64][((int)(64 / 8))];
  float cos_table[64][((int)(64 / 8))];
  float stepsizes[((int)(64 / 8))];
  float stepsizes_sqr[((int)(64 / 8))];
  int max_angular_steps_needed_for_quant_level[13];

  float decimated_weights[2 * 87 * 64];
  uint8_t u8_quantized_decimated_quantized_weights[2 * 2048 * 64];
  float decimated_quantized_weights[2 * 87 * 64];
  float flt_quantized_decimated_quantized_weights[2 * 2048 * 64];

  error_weighting_params m_ewp;
  int m_compress_to_mono;
  block_size_descriptor bsd;
  float m_Quality;
  partition_info partition_tables[5][(1 << 10)];
} ASTC_Encode

    __attribute__((aligned))

    ;

typedef struct intintLCGQuadraticOpenCompactCLHash_Bucket {
  int key;
  int value;
} intintLCGQuadraticOpenCompactCLHash_Bucket;

typedef struct {

  DirectLightIlluminateInfo illumInfo;

  Seed seedPassThroughEvent;

  int throughShadowTransparency;
} GPUTaskDirectLight;

typedef struct {
  float RcpGridStepW;
  float RcpGridStepH;
  CFlux2DCalculatorPars m_CalcParams;

  int SpeedCacheMatrixOffset;
  int SpeedCacheMatrixWidthStep;

  int HeightMapOffset;
  int HeightMapWidthStep;
  int UMapOffset;
  int UMapWidthStep;
  int VMapOffset;
  int VMapWidthStep;

  int HeightMapOffsetOut;
  int HeightMapWidthStepOut;
  int UMapOffsetOut;
  int UMapWidthStepOut;
  int VMapOffsetOut;
  int VMapWidthStepOut;

  int HeightMapBottomOffset;
  int HeightMapBottomWidthStep;

  int PrecompDataMapBottomOffset;
  int PrecompDataMapBottomWidthStep;

  float Hmin;
  float gravity;
  float halfGravity;
  float tau_div_deltaw;
  float tau_div_deltah;
  float tau_div_deltaw_quarter;
  float tau_div_deltah_quarter;
  float tau_mul_Cf;
  float minNonZeroSpeed;

} CKernelData;

typedef struct _clbpt_ins_pkt {
  unsigned int target;
} clbpt_ins_pkt;

typedef struct androidpin_tmp {
  unsigned int digest_buf[5];

} androidpin_tmp_t;

typedef struct {
  int c00;
  int c01;
} scan_t;

typedef struct {
  type_simbolo simbolo;
  int num_escolhas;
  t_escolha escolhas[20];
} t_regra;

typedef struct RGBcolor_struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char pad;
} RGBcolor;

typedef struct primecoinInput {
  primecoinBlockcl_t blocks;

  uint32_t nPrimorialHashFactor;
  uint32_t nHashFactor;
  uint32_t nPrimorialMultiplier;
  uint32_t nOverrideTargetValue;
  uint32_t nOverrideBTTargetValue;
  uint32_t lSieveTarget;
  uint32_t lSieveBTTarget;

  uint32_t nPrimes;
  uint32_t primeSeq;

  mpzcls_t mpzPrimorial;
  mpzcls_t mpzFixedMultiplier;
} primecoinInput_t;

typedef struct {
  unsigned int texIndex;
  unsigned int channelIndex;
} SplitFloat3TexParam;

typedef struct _clbpt_del_pkt {
  unsigned int target;
} clbpt_del_pkt;

typedef struct {
  cl_uint length;
  cl_uint ncols;
  cl_uint nslots;
  cl_char is_outer;
  cl_char __padding__[3];
} kern_hashtable;

typedef struct {
  int cells[1024];
} Mat32X32;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_R __attribute__((aligned(4)));
  unsigned int idx_anisoX;
  unsigned int idx_anisoY;
} NewWardElem;

typedef struct {
  float orig_data[864];
  float work_data[864];
  float deriv_data[864];

  uint8_t rgb_lns[864];
  uint8_t alpha_lns[864];
  uint8_t nan_texel[864];

  float red_min, red_max;
  float green_min, green_max;
  float blue_min, blue_max;
  float alpha_min, alpha_max;
  int grayscale;
  int xpos, ypos, zpos;
} imageblock;

typedef struct tc64_tmp {
  unsigned long long ipad[8];
  unsigned long long opad[8];

  unsigned long long dgst[32];
  unsigned long long out[32];

} tc64_tmp_t;

typedef struct {
  float scaleX;
  float scaleY;
  float dx;
  float dy;
  float angleX;
  float angleY;
} rectangleData;

typedef struct {
  int m_numChildShapes;
  int blaat2;
  int m_shapeType;
  int m_shapeIndex;

} btCollidableGpu;

typedef struct clb2Impulse {
  float normalImpulse1;
  float tangentImpulse1;

  float normalImpulse2;
  float tangentImpulse2;
} clb2Impulse;

typedef struct {
  unsigned int vertsOffset;
  unsigned int trisOffset;

  float trans[4][4];
  float invTrans[4][4];
} Mesh;

typedef struct CvPyramid {
  uchar **ptr;
  double *rate;
  int *step;
  uchar *state;
  int level;
} CvPyramid;

typedef struct ILpointf {
  float x;
  float y;
} ILpointf;

typedef struct wpa {
  unsigned int pke[25];
  unsigned int eapol[64 + 16];
  unsigned short eapol_len;
  unsigned char message_pair;
  int message_pair_chgd;
  unsigned char keyver;
  unsigned char orig_mac_ap[6];
  unsigned char orig_mac_sta[6];
  unsigned char orig_nonce_ap[32];
  unsigned char orig_nonce_sta[32];
  unsigned char essid_len;
  unsigned char essid[32];
  unsigned int keymic[4];
  unsigned int hash[4];
  int nonce_compare;
  int nonce_error_corrections;

} wpa_t;

typedef struct {
  constant int *array;
  size_t size;
} direction_array_data;

typedef struct {
  int nposi, nposj;
  int nmaxpos;
  float fmaxscore;
  int noutputlen;
} MAX_INFO;

typedef struct __GSSpriteout GSSpriteOut;

__constant float4 g_positions[4] = {
    (float4)(-1.0f, 1.0f, 0.0f, 0.0f), (float4)(1.0f, 1.0f, 0.0f, 0.0f),
    (float4)(-1.0f, -1.0f, 0.0f, 0.0f), (float4)(1.0f, -1.0f, 0.0f, 0.0f)};

typedef struct {
  bool hit;
  Ray ray;
  float t;
  float3 pos;
  float3 texcoord;
  float3 normal;
  const __global Triangle *object;
} IntersectData;

typedef struct {
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue;
  cl_uint row_alignment;
  cl_ulong max_device_work_group_size;

} CvGpuDevice;

typedef struct DecodeVariables dvars;

struct BufferVariables {

  int OutData_image_width;
  int OutData_image_height;
  int OutData_comp_vpos[3];
  int OutData_comp_hpos[3];
  unsigned char OutData_comp_buf[3][(90 * 59)];
  unsigned char *CurHuffReadBuf;
  unsigned int CurHuffReadBufPtr;
};

typedef struct {
  float2 p1;
  float2 p2;
} wall_t;

typedef struct {
  unsigned int texMapIndex;
  float shiftU, shiftV;
  float scaleU, scaleV;
} TexMapInstance;

typedef struct dcc2_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[5];
  unsigned int out[4];

} dcc2_tmp_t;

typedef struct potential_sols_s {
  unsigned int nr;
  unsigned int values[4096][2];
} potential_sols_t;

typedef struct {
  void *address;
  struct {
    bool input;
    bool output;
  } flags;
  short dimension_count;

  nanos_region_dimension_internal_t const *dimensions;

  ptrdiff_t offset;
} nanos_copy_data_internal_t;

typedef struct {
  S1 s1;
  int a;
  S1 s2;
} S2;

typedef struct {
  TextureData data;
} Texture;

typedef struct {
  unsigned int length;
  uchar salt[256];
} keystore_salt;

typedef struct {

  float x;
  float y;
  float t;

  float d1;
  float d2;
  float d3;

  float wz;
  float rayZ[20];
} ClParticle;

typedef struct {
  float _0;
  Tuple_float_int_int _1;
} Tuple_float_Tuple_float_int_int;

typedef struct {
  union {
    struct {

      bbox bounds[2];
    };

    struct {

      int i0, i1, i2;

      int child0;

      int shape_mask;

      int shape_id;

      int prim_id;

      int child1;
    };
  };

} bvh_node;

typedef struct {
  long start;
  long end;
} Range;

typedef struct {
  Range range;
  int uniqueElements;
  int subsetSize;
} RangeCombinatoricContext;

typedef struct {
  OutputWidthVector data[16];
} PoolOutput;

typedef struct {
  cl_image_format pixelFormat;
  cl_mem_flags supportedFlags;
  cl_bool supports2D;
  cl_bool supports3D;
} clu_image_format;

typedef struct {
  bool input;
  bool output;
  bool can_rename;
  bool concurrent;
  bool commutative;
} nanos_access_type_internal_t;

typedef struct {

  void *address;

  nanos_access_type_internal_t flags;

  short dimension_count;
  nanos_region_dimension_internal_t const *dimensions;

  ptrdiff_t offset;
} nanos_data_access_internal_t;

typedef struct rar3_tmp {
  unsigned int dgst[17][5];

} rar3_tmp_t;

typedef struct tc {
  unsigned int salt_buf[32];
  unsigned int data_buf[112];
  unsigned int keyfile_buf[16];
  unsigned int signature;

} tc_t;

typedef struct {
  unsigned int amountTexIndex, tex1Index, tex2Index;
} MixTexParam;
typedef struct {
  unsigned int dataIndex;
} UVMapping3DParam;

typedef struct aes_key_st {
  unsigned int rd_key[4 * (14 + 1)];
  int rounds;
} AES_KEY;

typedef struct MP_GridData {

  int size[3];
  double element[3];
  int ntot;
  int ntype;
  short *type;
  short *update;
  double *val;
  double *buf;
  double *cx, *cy, *cz;
  int *inter_x, *inter_y, *inter_z;
  double *coef_x, *coef_y, *coef_z;
  double *rhoc;
  int bound[6];
  int step;
  long rand_seed;
  int local_coef;
} MP_GridData;

typedef struct TINYMT32J_T {
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
  unsigned int s3;
} tinymt32j_t;

typedef struct {
  int x;
  int y;
  int z;
} Int3;

typedef struct sieveTemp {
  uint32_t vfCC1[((1024 * 1024 * 2) / 32) * (10 + 10)];
  uint32_t vfCC2[((1024 * 1024 * 2) / 32) * (10 + 10)];

  uint32_t vfExtBitwin[((1024 * 1024 * 2) / 32) * (10 + 1)];
  uint32_t vfExtCC1[((1024 * 1024 * 2) / 32) * (10 + 1)];
  uint32_t vfExtCC2[((1024 * 1024 * 2) / 32) * (10 + 1)];

  mpzcl_t mpzFixedFactor;
  mpzcl_t mpzQuot;

  uint32_t nFixedFactorCombinedMod[1024];
  uint32_t nPrimeCombined[1024];
  uint32_t nFixedInverse[((15360)) * 1024];

  uint32_t flatFixedInverse[((15360)) * 1024];
  uint32_t flatCC1Multipliers[((15360))];
  uint32_t flatCC2Multipliers[((15360))];

  result_t bitwins;
  result_t cc1s;
  result_t cc2s;

  uint32_t nSievesComplete;
} sieveTemp_t;

typedef struct {
  double speeds[9];
} t_speed;

typedef struct {
  int isCollide;
  float normal[2];
  float fraction;
  unsigned int shapeIndex;
} b2clRayCastOutput;

typedef struct {
  float k;
  float ior;

} RT_BTDF;

typedef struct list_t {
  uint32_t length;
  int *compare;
  void *datum_delete;
} list_t;

typedef struct {
  float gain_r, gain_g, gain_b;
} AreaLightParam;

typedef struct {
  ulong s[17];
  char p1, p2;
} lfib_state;

typedef struct {
  int integer;
  float4 float_vec;
} InnerStruct;

typedef struct {
  InnerStruct PrivateStruct;
  char another_field;
} OuterStruct;

typedef struct {
  float3 a;
  float3 b;
} line3;

typedef struct {
  Ray r[16];
  int depth[16];
  float3 weight[16];
  int top;
} Rstack;

typedef struct {
  t_item_programa programa[128];
} t_prog;

typedef struct {

  salt_t pbkdf2;

  union {
    uint64_t qword[32 / 8];
    uint8_t chr[32];
  } blob;
} bitwarden_salt_t;

typedef struct {
  unsigned int lightSceneIndex;
  unsigned int lightID;

  int visibility;
  int isDirectLightSamplingEnabled;

  union {
    NotIntersectableLightSource notIntersectable;
    TriangleLightParam triangle;
  };
} LightSource;

typedef struct _gcidxinfo gcidxinfo;

struct _gcidxinfo {
  unsigned int num_h2_leafs;

  unsigned int idx_off;

  unsigned int ridx_size;

  unsigned int ridx_off;

  global const unsigned int *ridx;

  unsigned int yt_off;

  global float *yt;

  local float *ytl;

  unsigned int cidx_size;

  unsigned int cidx_off;

  global const unsigned int *cidx;

  unsigned int xt_off;

  global const float *xt;
};

typedef struct _World {
  unsigned int numSpheres;
  Sphere spheres[10];

  unsigned int numPlanes;
  Plane planes[10];

  Camera camera;
} World;

typedef struct {
  Spectrum rgb;
} BlackBodyParam;

typedef struct __blake2bp_state {
  blake2b_state S[4][1];
  blake2b_state R[1];
  uint8_t buf[4 * 16];
  size_t buflen;
} blake2bp_state;

typedef struct IntersectPoint {

  float3 minT;

  float3 maxT;
} IntersectPoint;

typedef struct __attribute__((aligned(4))) {
  unsigned int idx_envLightProperty;
} EnvironmentHead;

typedef struct {
  float level_gr;
  float level_r;
  float level_b;
  float level_gb;
} BLCConfig;

typedef struct {
  int a;
  int b;
  float restLength;
  float strength;
} Spring;

typedef struct {
  unsigned int length;
  unsigned int count;
  uchar salt[8];
} gpg_salt;

typedef struct _cl_mem_android_native_buffer_host_ptr {

  cl_mem_ext_host_ptr ext_host_ptr;

  void *anb_ptr;

} cl_mem_android_native_buffer_host_ptr;

typedef struct {
  Transform appliedTrans;
  int appliedTransSwapsHandedness;
} TriangleMeshParam;

typedef struct {
  unsigned int length;
  ushort v[255];
} sevenzip_password;

typedef struct s_quad {
  float a;
  float b;
  float c;
  float d;
  float res;
} t_quad;

typedef struct {
  void *buf;
  int dim[2];
} __PSGrid2DDev;

typedef struct b2clRopeJointData {
  float localAnchorA[2];
  float localAnchorB[2];
  float maxLength;
  float nlength;
  float u[2];
  float rA[2];
  float rB[2];
  float localCenterA[2];
  float localCenterB[2];
  float invMassA;
  float invMassB;
  float invIA;
  float invIB;
  float mass;
  int limitState;
} b2clRopeJointData;

typedef struct clJoint {
  int index;

  union a1 {
    struct x1 {
      float impulse[3];
    } x;
    struct y1 {
      float scalarImpulse;
      float springImpulse;
    } y;
  } a;
  float motorImpulse;

  int float3;
  int type;
  int collideConnected;

  int indexA, indexB, indexC, indexD;

  union b1 {
    b2clDistanceJointData distanceJointData;
    b2clRevoluteJointData revoluteJointData;
    b2clPrismaticJointData prismaticJointData;
    b2clGearJointData gearJointData;
    b2clPulleyJointData pulleyJointData;
    b2clRopeJointData ropeJointData;
    b2clWheelJointData wheelJointData;
    b2clWeldJointData weldJointData;
    b2clMouseJointData mouseJointData;
    b2clFrictionJointData frictionJointData;
  } b;
} b2clJoint;

typedef struct half13 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
} half13;

typedef struct {
  float3 p;

  float3 s_n;

  float r;

  RT_Mat4f invMatrix;

  RT_BBox bbox;
} RT_Primitive;

typedef struct Header_ {
  char magic_key[4];
  int whole_file_size;
  int header_size;
  int binary_size;
  int signature_size;
} Header;

typedef struct s_txdata {
  unsigned int width;
  unsigned int height;
  unsigned int start;
} t_txdata;

typedef struct {
  unsigned int kdTexIndex;
  unsigned int ktTexIndex;
  unsigned int ksTexIndex;
  unsigned int ksbfTexIndex;
  unsigned int nuTexIndex;
  unsigned int nubfTexIndex;
  unsigned int nvTexIndex;
  unsigned int nvbfTexIndex;
  unsigned int kaTexIndex;
  unsigned int kabfTexIndex;
  unsigned int depthTexIndex;
  unsigned int depthbfTexIndex;
  unsigned int indexTexIndex;
  unsigned int indexbfTexIndex;
  int multibounce;
  int multibouncebf;
} GlossyTranslucentParam;

typedef struct {
  unsigned int sigmaATexIndex;
  unsigned int sigmaSTexIndex;
  unsigned int gTexIndex;
  float stepSize;
  unsigned int maxStepsCount;
  int multiScattering;
} HeterogenousVolumeParam;

typedef struct {
  unsigned int iorTexIndex;

  unsigned int volumeEmissionTexIndex;
  unsigned int volumeLightID;
  int priority;

  union {
    ClearVolumeParam clear;
    HomogenousVolumeParam homogenous;
    HeterogenousVolumeParam heterogenous;
  };
} VolumeParam;

typedef struct {
  float3 eyePos;
  float4 mcPos;
  float3 mcNormal;
  float2 pixelPos;
  float seed;
} TRenderState;

typedef struct {
  float data[256];
} TypedStruct;

typedef struct AmbientLight {
  float4 float3;
  float intensity;
  int enableAmbientOcclusion;
  int ambientOcclusionSampleCountSqrt;
  float ambientOcclusionMaxSampleDistance;
  float ambientOcclusionSampleDistribution;
} AmbientLight;

typedef struct blake2s_param_t {
  uchar digest_length;
  uchar key_length;
  uchar fanout;
  uchar depth;
  unsigned int leaf_length;
  uchar node_offset[6];
  uchar node_depth;
  uchar inner_length;
  uchar salt[8];
  uchar personal[8];
} blake2s_param;

typedef struct {
  uint32_t iterations;
  uint32_t key_len;
  uint32_t length;
  uint8_t salt[64];
  uint32_t comp_len;
  uchar passverify[2];
} zip_salt;

typedef struct {
  float4 m[4];
} TMatrix;

typedef struct Mat13 {
  float data[3];
} Mat13;

typedef struct {
  TMatrix InvWorldTransform;
  TMatrix WorldTransform;
  TMaterial material;
  float radius;
  unsigned int index;
} TSphere;

typedef struct s_queue {
  int count;
  t_lst *first;
  t_lst *last;
} t_queue;

typedef struct b2clFilter {
  unsigned short categoryBits;
  unsigned short maskBits;
  short groupIndex;
  short dummy;
} b2clFilter;

typedef struct b2clPolygonShape {
  float2 m_centroid;
  float2 m_vertices[8];
  float2 m_normals[8];
  int m_type;
  float m_radius;
  int m_vertexCount;
  int m_bIsSensor;
  b2clFilter m_filter;
} b2clPolygonShape;

typedef struct {
  float3 pMin;
  float3 pMax;
} BBox;

typedef struct _bvh_node_t {
  float bbox_min[3];
  union {
    unsigned int prim_index;
    unsigned int left_child;
  };
  float bbox_max[3];
  union {
    unsigned int prim_count;
    unsigned int right_child;
  };

} bvh_node_t;

typedef struct bitlocker {
  unsigned int type;
  unsigned int iv[4];
  unsigned int data[15];
  unsigned int wb_ke_pc[0x100000][48];

} bitlocker_t;

typedef struct {
  unsigned int width;
  unsigned int height;
} TImage;

typedef struct {
  cl_device_type devType;
  cl_uint vendor_id;

  cl_uint max_compute_units;
  cl_uint max_work_item_dims;
  size_t max_work_items_per_dimension[3];
  size_t max_wg_size;

  cl_uint device_preferred_vector_width_char;
  cl_uint device_preferred_vector_width_short;
  cl_uint device_preferred_vector_width_int;
  cl_uint device_preferred_vector_width_long;
  cl_uint device_preferred_vector_width_float;
  cl_uint device_preferred_vector_width_half;

  cl_uint device_native_vector_width_char;
  cl_uint device_native_vector_width_short;
  cl_uint device_native_vector_width_int;
  cl_uint device_native_vector_width_long;
  cl_uint device_native_vector_width_float;
  cl_uint device_native_vector_width_half;

  cl_uint max_clock_frequency;
  cl_uint address_bits;
  cl_ulong max_mem_alloc_size;

  cl_bool image_support;
  cl_uint max_read_image_args;
  cl_uint max_write_image_args;
  size_t image2d_max_width;
  size_t image2d_max_height;
  size_t image3d_max_width;
  size_t image3d_max_height;
  size_t image3d_max_depth;
  cl_uint max_samplers;

  size_t max_parameter_size;
  cl_uint mem_base_addr_align;
  cl_uint min_data_type_align_size;
  cl_device_fp_config single_fp_config;
  cl_device_mem_cache_type global_mem_cache_type;
  cl_uint global_mem_cacheline_size;
  cl_ulong global_mem_cache_size;
  cl_ulong global_mem_size;
  cl_ulong constant_buffer_size;
  cl_uint max_constant_args;
  cl_device_local_mem_type local_mem_type;
  cl_ulong local_mem_size;
  cl_bool error_correction_support;

  cl_bool unified_memory;
  size_t profiling_timer_resolution;
  cl_bool endian_little;
  cl_bool device_available;
  cl_bool compiler_available;
  cl_device_exec_capabilities device_capabilities;
  cl_command_queue_properties queue_properties;
  cl_platform_id platform_id;

  char device_name[256];
  char device_vendor[256];
  char driver_version[256];
  char device_profile[256];
  char device_version[256];
  char opencl_c_version[256];
  char extensions[(256 * 128)];
} clu_device_info;

typedef struct {
  float dVal;
  unsigned int ktIndex;
} ColorDepthTexParam;

typedef struct {
  unsigned int ridptr;
  unsigned int rid;
} match_t;

typedef struct {
  float x;
  float y;
  float z;
} Single3;

typedef struct {
  const int width;
  const int height;
  global const float *pdfTable;
  global const float *probTable;
  global const int *aliasTable;
} EnvMapContext;

typedef struct {
  unsigned int id;
  transform_t previous_transform;
  float2 wheels_angular_speed;
  unsigned int front_led;
  unsigned int rear_led;
  unsigned int collision;
  float energy;
  float fitness;
  int last_target_area;
  unsigned int entered_new_target_area;

  float sensors[13];
  float actuators[4];
  float hidden[3];

  char raycast_table[12 + 2];
} robot_t;

typedef struct {
  unsigned int id;

  robot_t robots[64];

  float arena_height;
  float arena_width;

  wall_t walls[4];
  target_area_t target_areas[2];

  float weights[4][13 + 3];
  float bias[4];

  float weights_hidden[3][13];
  float bias_hidden[3];
  float timec_hidden[3];

  unsigned int random_offset;
} world_t;

typedef struct {
  int x;
  int y;
} point;

typedef struct {
  unsigned int iterations;
  comp z0;
  point size;
  comp vp_ul;
  comp vp_dr;
  float float3;
} t_fractol_args;

typedef struct {
  float len;
  float arg;
} ComplexPolar;

typedef struct {
  sRayMarchingOut rayMarchingOut;
  float3 point;
  float4 resultShader;
  float3 objectColour;
  float3 normal;
  float fogOpacity;
  bool found;
} sRayRecursionOut;

typedef struct {
  int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_zswap_params_t;

typedef struct struct_of_structs_arg {
  int i1;
  float f1;
  struct_arg_t s1;
  int i2;
} struct_of_structs_arg_t;

typedef struct {
  union {
    ulong x;
    uint2 x2;
  };
  ulong w;
} msws_state;

typedef struct int_pair {
  long a;
  long b;
} int_pair;

typedef struct {
  float3 o;
  float3 at;
  float3 up;
  float fov;
} Cam;

typedef struct {
  int2 s0, s1, s2, s3;
} v4int2;

typedef struct electrum {
  secp256k1_t coords;

  unsigned int data_buf[256];

} electrum_t;

typedef struct {
  float4 Position;
  float4 Color;
  float Size;

  ParticleForce Force;

} NodeInstance;

typedef struct QueueRecord {
  JobDescription *Array;
  int Capacity;
  int Rear;
  int Front;
  int ReadLock;
} QueueRecord;

typedef struct s_context {
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
} t_context;

typedef struct {
  int state;
  float t;
} b2clTOIOutput;

typedef struct {
  unsigned int i[8];

  unsigned int pw_len;

} comb_t;

typedef struct scrypt_hash_state_t {
  uint4 state4[(1600 + 127) / 128];
  uint4 buffer4[(72 + 15) / 16];

} scrypt_hash_state;

typedef struct {

  RT_ViewPlane vp;
  float3 background;
  int numLights;
  int numLamps;
  int numObjects;
  int numSamples;
  int numSets;
  unsigned int seed;
} RT_DataScene;

typedef struct {
  union {
    int8 vecInt;
    int ints[8];
  };
} Int8CastType;

typedef struct {
  unsigned int size_bytes;
  char password[(64 - sizeof(unsigned int))];
} password_t;

typedef struct {
  float2 uv;
  const __global float *invTransform;
  int triangleIndex;
  float t;
  bool hit;
} ShadingData;

typedef struct __attribute__((packed)) _Gaussian {

  float w;

  float2 mu;

  float2 var;
} Gaussian;

typedef struct {
  float4 float3;
  float depth;
} RayResult;

typedef struct KeyInfo {
  unsigned char partialKeyAndRandomBytes[10];
  unsigned char expansioinFunction[96];
} KeyInfo;

typedef struct Foo {
  int *ptrField;
} Foo;

typedef struct PerspectiveMotionTransform {
  Transform pre;
  Transform post;
} PerspectiveMotionTransform;

typedef struct {
  void *buf;
  int dim[3];
} __PSGrid3DDev;

typedef struct {
  float4 amod;
  float4 bs;
  float4 bis;
} processed_line4;

typedef struct {
  float3 p;
  float3 a;
  float3 b;
  float3 normal;
  float r;
  RT_Emissive material;

} RT_Lamp;

typedef struct keepass_tmp {
  unsigned int tmp_digest[8];

} keepass_tmp_t;

typedef struct {
  int m_nContacts;
  int m_staticIdx;
  float m_scale;
  int m_nSplit;
} ConstBufferSSD;

typedef struct _ant {
  int posision, lastposision;
  unsigned char colony;
} ant;

typedef struct _VolumeSlicingParams {

  float3 view;

  float3 verts[8];

  float3 tVerts[8];

  float dPlaneStart;

  float dPlaneIncr;

  int frontIdx;

  int numSlices;
} VolumeSlicingParams;

typedef struct TargetDevice {
  cl_device_id id;
  bool hwInfoValid;
  DeviceHwInfo hwInfo;
} TargetDevice;

typedef struct atmi_kernel_s {

  uint64_t handle;
} atmi_kernel_t;

typedef struct {
  unsigned int krTexIndex;
  unsigned int ktTexIndex;
  unsigned int exteriorIorTexIndex, interiorIorTexIndex;
  unsigned int nuTexIndex;
  unsigned int nvTexIndex;
} RoughGlassParam;

typedef struct iwork_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[5];
  unsigned int out[5];

} iwork_tmp_t;

typedef struct digest {
  unsigned int digest_buf[4];

} digest_t;

typedef struct PointLight {
  float4 float3;
  float4 position;
  float intensity;
  float maxDistance;
  float attenuation;
  int enableAreaLight;
  int areaLightSampleCountSqrt;
  float areaLightRadius;
} PointLight;

typedef struct tree_node {

  int axis;
  float splitting_value;

} TREE_NODE;

typedef struct {
private
  char *p1;
  local char *p2;
  constant char *p3;
  global char *p4;
  char *p5;
} StructTy1;

typedef struct _camera_t {
  float4 origin, fwd;
  float4 side, up;
  int flags;
} camera_t;

typedef struct wrapper_s {
  char *_name;
  cl_program _program;
} wrapper_t;

typedef struct SmoothedPointPair {

  int2 p1;

  float2 p2;
} SmoothedPointPair;

typedef struct {
  global float *data;
  int3 size;
} volume;

typedef struct {
  float3 min_corner;
  int3 dimensions;
  float spacing;
} mesh_descriptor;

typedef struct {
  cl_uint next;
  cl_uint hash;
  cl_uint rowid;
  cl_uint t_len;
} kern_hashentry;

typedef struct sMatrix4 {
  float4 data[4];
} Matrix4;

typedef struct KernelTables {
  int beckmann_offset;
  int pad1, pad2, pad3;
} KernelTables;

typedef struct action {
  struct action *next;
  short symbol;
  short number;
  short prec;
  char action_code;
  char assoc;
  char suppressed;
} action;

typedef struct axcrypt_tmp {
  unsigned int KEK[4];
  unsigned int lsb[4];
  unsigned int cipher[4];

} axcrypt_tmp_t;

typedef struct {
  unsigned int tex1Index, tex2Index;
} GreaterThanTexParam;

typedef struct blake2b_ctx_vector {
  unsigned long long m[16];
  unsigned long long h[8];

  unsigned int len;

} blake2b_ctx_vector_t;

typedef struct box_str {

  int x, y, z;
  int number;
  long offset;

  int nn;
  nei_str nei[26];

} box_str;

typedef struct {
  RT_BBox bbox;
  int nx, ny, nz;
  int numTriangles;
  int numCells;

  RT_Mat4f invMatrix;
} RT_Grid;

typedef struct {
  int n, lda, j0, npivots;
  short ipiv[32];
} claswp_params_t2;

typedef struct {
  Vector p, d;
  unsigned int lightID;
  Spectrum alpha;
  Normal landingSurfaceNormal;
  int isVolume;
} Photon;

typedef struct {
  unsigned int v[4];
} phpass_hash;

typedef struct sieveOutput {
  bresult_t results;
  uint32_t nBitwins;
  uint32_t nCC1s;
  uint32_t nCC2s;
  uint32_t nLayer;
  uint32_t nTested;
} sieveOutput_t;

typedef struct KernelErrorInfo {
  unsigned int wrongArg;
} KernelErrorInfo;

typedef struct {
  unsigned int fresnelTexIndex;
  unsigned int nTexIndex;
  unsigned int kTexIndex;
  unsigned int nuTexIndex;
  unsigned int nvTexIndex;
} Metal2Param;

typedef struct drupal7_tmp {
  unsigned long long digest_buf[8];

} drupal7_tmp_t;

typedef struct {
  unsigned int tex1Index, tex2Index;
} DivideTexParam;

typedef struct {
  ulong f1;
  ulong f2;
} ConstantFP128Ty;

typedef struct s_argn {
  int2 screen_size;
  int nb_objects;
  int nb_lights;
  int map_primitives;
  float ambient;
  float direct;
  int antialias;
  int bounce_depth;
  int filter;
  int stereoscopy;
  t_texture skybox;
  int nb_info;
  int nb_materials;
} t_argn;

typedef struct KernelKey {
  cl_device_id device;
  cl_context context;
  unsigned int nrDims;
  SubproblemDim subdims[4];
} KernelKey;

typedef struct def_ClothEdge {
  unsigned int edgeID;
  float initialDihedralAngle;
  float initialLength;
} ClothEdgeData;

typedef struct {
  Spectrum n, k;
} FresnelConstParam;

typedef struct pw {
  unsigned int i[64];

  unsigned int pw_len;

} pw_t;

typedef struct {
  union {
    char c[16];
    unsigned int w[16 / 4];
  } v;
  int l;
} lotus5_key;

typedef struct {

  int type;

  size_t size;

  const char *id;

  const char *name;

  size_t npars;
} object;

typedef struct {
  float2 min_max;
  float2 min_max_output;
  float default_value;
  float dummy_for_alignment;
} FilterParameters;

typedef struct {

  options *opts;

  int *reqs;

  size_t nobjs;
  object *objs;
} input;

typedef struct {
  float atom_charges_const[8];
  int atom_types_const[8];
  int atom_types_map_const[8];
} kernelconstant_interintra;

typedef struct __attribute__((aligned(4))) {
  float uComponent;
  float uDir[2];
} BSDFSample;

typedef struct reductions reductions;
struct reductions {
  struct reductions *next;
  short number;
  short nreds;
  short rules[1];
};

typedef struct seven_zip_tmp {
  unsigned int h[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

} seven_zip_tmp_t;

typedef struct DirectionalLight {

  float3 direction;
} DirectionalLight;

typedef struct wpa_eapol {
  unsigned int pke[32];
  unsigned int eapol[64 + 16];
  unsigned short eapol_len;
  unsigned char message_pair;
  int message_pair_chgd;
  unsigned char keyver;
  unsigned char orig_mac_ap[6];
  unsigned char orig_mac_sta[6];
  unsigned char orig_nonce_ap[32];
  unsigned char orig_nonce_sta[32];
  unsigned char essid_len;
  unsigned char essid[32];
  unsigned int keymic[4];
  int nonce_compare;
  int nonce_error_corrections;
  int detected_le;
  int detected_be;

} wpa_eapol_t;

typedef struct {
  uint32_t length;
  buffer_32 pass[(24 / 4)];
} sha256_password;

typedef struct {
  float gain, uScale, vScale, uDelta, vDelta;
  float Du, Dv;

  unsigned int imageMapIndex;
} ImageMapInstanceParam;

typedef struct FuncTable {

  void *fillRandom;

  void *fillMarker;

  int *compare;

  void *mul;
} FuncTable;

typedef struct electrum_wallet {
  unsigned int salt_type;
  unsigned int iv[4];
  unsigned int encrypted[4];

} electrum_wallet_t;

typedef struct {
  unsigned int sse[15 + 1];
  int sum[15 + 1];
} rd_calc_buffers;

typedef struct {
  float4 positionRadius;
  float4 normalQuality;
} Splat;

typedef struct {
  float x, y;
} Sleef_float2;

typedef struct __attribute__((aligned(4))) {
  ProceduralTextureHead head;
  float v[2] __attribute__((aligned(4)));
} FloatCheckerBoardTexture;

typedef struct {
  float3 center;
  float radius;
  float diffuseness;
  float emittance;
  float reflectance;
  float transmittance;
  Texture texture;
} Object;

typedef struct Kstring {

  char buf[512];
} Kstring;

typedef struct {
  unsigned int m_dict_ofs, m_dict_avail, m_first_call, m_has_flushed;
  int m_window_bits;
  unsigned char m_dict[1];

} inflate_state;

typedef struct _texture_t {
  ushort width;
  ushort height;
  uchar page[128];
  ushort pos[128][2];
} texture_t;

typedef struct PointColor {
  float4 point;
  float4 float3;
} PointColor;

typedef struct {
  unsigned int baseTexIndex, exponentTexIndex;
} PowerTexParam;

typedef struct _QueueGlobal {
  unsigned int front;
  unsigned int back;
  int filllevel;
  unsigned int activeBlocks;
  unsigned int processedBlocks;
  int debug;
} QueueGlobal;

typedef struct IsoPoint {
  float x;
  float y;
  float z;
  float xn;
  float yn;
  float zn;
  float isoValue;
  float dummy;
} IsoPoint;

typedef struct TensorInfoCl {
  unsigned int sizes[128];
  unsigned int strides[128];
  int offset;
  int dims;
} TensorInfoCl;

typedef struct streebog256_ctx {
  unsigned long long h[8];
  unsigned long long s[8];
  unsigned long long n[8];

  unsigned int w0[4];
  unsigned int w1[4];
  unsigned int w2[4];
  unsigned int w3[4];

  int len;

  __constant unsigned long long (*s_sbob_sl64)[256];

} streebog256_ctx_t;

typedef struct scrypt_hmac_state_t {
  scrypt_hash_state inner;
  scrypt_hash_state outer;
} scrypt_hmac_state;

typedef struct __attribute__((aligned(16))) {
  FresnelHead head;
  float3 eta __attribute__((aligned(16))), k;
} FresnelConductor;

typedef struct {
  salt_t pbkdf2;
  uint32_t bloblen;
  uint8_t blob[64];
} ansible_salt_t;

typedef struct intintLCGQuadraticOpenCompactCLHash_TableData {
  int hashID;
  unsigned int numBuckets;
  intintHash_CompressLCGData compressFuncData;
} intintLCGQuadraticOpenCompactCLHash_TableData;

typedef struct {
  unsigned int sampleCount;
} GPUTaskStats;

typedef struct s_basis {
  float3 u;
  float3 v;
  float3 w;
} t_basis;

typedef struct {
  long a, b, c, d;
private
  char *p;
} StructTy3;

typedef struct __attribute__((aligned(4))) {
  uchar id;
  unsigned int idx_R __attribute__((aligned(4)));
  unsigned int idx_sigma;
} DiffuseRElem;

typedef struct {
  unsigned int x10, x11, x12, x20, x21, x22;
} mrg31k3p_state;

typedef struct {
  float diff_r, diff_g, diff_b;
  float refl_r, refl_g, refl_b;
  float exponent;
  float R0;
  int specularBounce;
} AlloyParam;

typedef struct streebog512_hmac_ctx_vector {
  streebog512_ctx_vector_t ipad;
  streebog512_ctx_vector_t opad;

} streebog512_hmac_ctx_vector_t;

typedef struct {
  unsigned int objStartIndex;
  unsigned int objSize;
} TBoxLink;

typedef struct {
  float3 amb;
  float3 diff;
  float3 spec;
  float3 mirror;
  float3 transp;
  float eta;
  int phong;
  enum { M_PHONG, M_MIRROR, M_DIELECTRIC } type;
} Matl;

typedef struct sphere_t {
  float3 center;
  float radius;
} sphere_t;

typedef struct _PAPhysicsHarmonicData {
  float strength, damping, restLength, noise;
  PAPhysicsFalloffData falloff;
} PAPhysicsHarmonicData;

typedef struct {
  float4 vertex;
  float4 float3;
} VertexAndColor;

typedef struct {
  union {
    float4 m_min;
    float m_minElems[4];
    int m_minIndices[4];
  };
  union {
    float4 m_max;
    float m_maxElems[4];
    int m_maxIndices[4];
  };
} b3AabbCL;

typedef struct STACK_ITEM {
  unsigned int offset;
  float hit;
  Box cube;
} STACK_ITEM;

typedef struct bestcrypt {
  unsigned int data[24];

} bestcrypt_t;

typedef struct __attribute__((aligned(16))) {
  uchar texType;
  float3 value __attribute__((aligned(16)));
} Float3ConstantTexture;

typedef struct DMatch {
  int queryIdx;
  int trainIdx;
  float dist;
} DMatch_t;

typedef struct win8phone {
  unsigned int salt_buf[32];

} win8phone_t;

typedef struct {
  uint2 x;
  uint2 c;
} mwc64xvec2_state_t;

typedef struct {
  cl_uint max_kernel_work_group_size;
} CvGpuKernel;

typedef struct {
  CvGpuKernel *_kernel;
  CvGpuMat *matrix_A;
  CvGpuMat *matrix_B;
  CvGpuMat *matrix_C;
} CvGpuGEMMKernel;

typedef struct {
  int updates;
} ComputeContext;

typedef struct CvNArrayIterator {
  int count;
  int dims;
  uchar *ptr[10];
  int stack[256];

} CvNArrayIterator;

typedef struct {
  buffer_64 alt_result[8];
  buffer_64 temp_result[(16 / 8)];
  buffer_64 p_sequence[((23 + 7) / 8)];
} sha512_buffers;

typedef struct streebog256_hmac_ctx {
  streebog256_ctx_t ipad;
  streebog256_ctx_t opad;

} streebog256_hmac_ctx_t;

typedef struct {
  int idum;
  int idum2;
  int iy;
  int iv[32];
} ran2_state;

typedef struct {
  float2 v[2];
} mat2t;

typedef struct {
  int n_parts;
  int n;
  int ldb;
  int parts[4];
  size_t part_pack_size[4];
  unsigned pack_part[4];
  size_t offset_compensation;
  size_t size;
  char reserved[200];
} mkldnn_rnn_packed_desc_t;

typedef struct {
  float4 position;
  float4 normal;
} Vertex;

typedef struct {
  int bodyIndex;
  float posX;
  float posY;
  float posAngle;
  float xfX;
  float xfY;
  float xfS;
  float xfC;
  float alpha;
  float velocityX;
  float velocityY;
  float velocityAngular;
} clb2SDBody;

typedef struct _centroid_t {
  data_type wgtCent;
  int sum_sq;
  unsigned int count;
} centroid_t;

typedef struct {
  CPTypedef pstruct;
} CPPTypedef;

typedef struct s_cylinder {
  float3 pos;
  float r;
  float h;
  float tex_scale;
} t_cylinder;

typedef struct {
  PrimitiveStruct primitive_struct;
  PrimitiveStruct *primitive_struct_pointer;
  PrimitiveStruct primitive_struct_array[1];
  PrivatePrimitivePointerStruct primitive_pointer_struct;
  PrivatePrimitivePointerStruct *primitive_pointer_struct_pointer;
  PrivatePrimitivePointerStruct primitive_pointer_struct_array[1];
  ArrayStruct array_struct;
  ArrayStruct *array_struct_pointer;
  ArrayStruct array_struct_array[1];
  PrivateArrayPointerStruct array_pointer_struct;
  PrivateArrayPointerStruct *array_pointer_struct_pointer;
  PrivateArrayPointerStruct array_pointer_struct_array[1];
} PrivateMainStruct;

typedef struct {
  float r;
  float g;
  float b;
} pixel_t;

typedef struct apple_secure_notes {
  unsigned int Z_PK;
  unsigned int ZCRYPTOITERATIONCOUNT;
  unsigned int ZCRYPTOSALT[16];
  unsigned int ZCRYPTOWRAPPEDKEY[16];

} apple_secure_notes_t;

typedef struct clb2Points {
  float2 rA1;
  float2 rB1;
  float normalMass1;
  float tangentMass1;

  float2 rA2;
  float2 rB2;
  float normalMass2;
  float tangentMass2;

  float velocityBias1;
  float velocityBias2;
} clb2Points;

typedef struct {
  float4 m_pos;
  float4 m_quat;
  float4 m_linVel;
  float4 m_angVel;

  unsigned int m_shapeIdx;
  float m_invMass;
  float m_restituitionCoeff;
  float m_frictionCoeff;
} Body;

typedef struct {
  int nthreads;
  int n_nuclides;
  int lookups;
  int avg_n_poles;
  int avg_n_windows;
  int numL;
  int doppler;
  int particles;
  int simulation_method;
  int kernel_id;
} Input;

typedef struct {
  salt_t pbkdf2;
  unsigned int mnemonic_length;
  uchar mnemonic[128];
} tezos_salt_t;

typedef struct nested_single_element_struct_arg {
  single_element_struct_arg_t i;
} nested_single_element_struct_arg_t;

typedef struct {
  unsigned int KdTexIndex;
  unsigned int Ks1TexIndex;
  unsigned int Ks2TexIndex;
  unsigned int Ks3TexIndex;
  unsigned int M1TexIndex;
  unsigned int M2TexIndex;
  unsigned int M3TexIndex;
  unsigned int R1TexIndex;
  unsigned int R2TexIndex;
  unsigned int R3TexIndex;
  unsigned int KaTexIndex;
  unsigned int depthTexIndex;
} CarPaintParam;

typedef struct _IntersectionResult {
  bool intersect;
  float u;
  float v;
  float t;
} IntersectionResult;

typedef struct float13 {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
} float13;

typedef struct pwsafe2_tmp {
  unsigned int digest[2];

  unsigned int P[18];

  unsigned int S0[256];
  unsigned int S1[256];
  unsigned int S2[256];
  unsigned int S3[256];

} pwsafe2_tmp_t;

typedef struct {
  PathState state;
  unsigned int depth;

  Spectrum throughput;
  BSDF bsdf;
} PathStateBase;

typedef struct params_common {
  int common_change_mem;
  int common_mem;
  int unique_mem;

  int frames_processed;

  int sSize;
  int tSize;
  int maxMove;
  float alpha;

  int no_frames;
  int frame_rows;
  int frame_cols;
  int frame_elem;
  int frame_mem;

  int endoPoints;
  int endo_mem;

  int epiPoints;
  int epi_mem;

  int allPoints;
  int in_rows;
  int in_cols;
  int in_elem;
  int in_mem;

  int in_pointer_mem;

  int in2_rows;
  int in2_cols;
  int in2_elem;
  int in2_mem;

  int conv_rows;
  int conv_cols;
  int conv_elem;
  int conv_mem;
  int ioffset;
  int joffset;
  int in2_pad_add_rows;
  int in2_pad_add_cols;
  int in2_pad_cumv_rows;
  int in2_pad_cumv_cols;
  int in2_pad_cumv_elem;
  int in2_pad_cumv_mem;

  int in2_pad_cumv_sel_rows;
  int in2_pad_cumv_sel_cols;
  int in2_pad_cumv_sel_elem;
  int in2_pad_cumv_sel_mem;
  int in2_pad_cumv_sel_rowlow;
  int in2_pad_cumv_sel_rowhig;
  int in2_pad_cumv_sel_collow;
  int in2_pad_cumv_sel_colhig;

  int in2_pad_cumv_sel2_rowlow;
  int in2_pad_cumv_sel2_rowhig;
  int in2_pad_cumv_sel2_collow;
  int in2_pad_cumv_sel2_colhig;
  int in2_sub_cumh_rows;
  int in2_sub_cumh_cols;
  int in2_sub_cumh_elem;
  int in2_sub_cumh_mem;

  int in2_sub_cumh_sel_rows;
  int in2_sub_cumh_sel_cols;
  int in2_sub_cumh_sel_elem;
  int in2_sub_cumh_sel_mem;
  int in2_sub_cumh_sel_rowlow;
  int in2_sub_cumh_sel_rowhig;
  int in2_sub_cumh_sel_collow;
  int in2_sub_cumh_sel_colhig;

  int in2_sub_cumh_sel2_rowlow;
  int in2_sub_cumh_sel2_rowhig;
  int in2_sub_cumh_sel2_collow;
  int in2_sub_cumh_sel2_colhig;
  int in2_sub2_rows;
  int in2_sub2_cols;
  int in2_sub2_elem;
  int in2_sub2_mem;
  int in2_sqr_rows;
  int in2_sqr_cols;
  int in2_sqr_elem;
  int in2_sqr_mem;

  int in2_sqr_sub2_rows;
  int in2_sqr_sub2_cols;
  int in2_sqr_sub2_elem;
  int in2_sqr_sub2_mem;
  int in_sqr_rows;
  int in_sqr_cols;
  int in_sqr_elem;
  int in_sqr_mem;

  int tMask_rows;
  int tMask_cols;
  int tMask_elem;
  int tMask_mem;

  int mask_rows;
  int mask_cols;
  int mask_elem;
  int mask_mem;

  int mask_conv_rows;
  int mask_conv_cols;
  int mask_conv_elem;
  int mask_conv_mem;
  int mask_conv_ioffset;
  int mask_conv_joffset;

} params_common;

typedef struct {
  float4 error_weights[216];
  float texel_weight[216];
  float texel_weight_gba[216];
  float texel_weight_rba[216];
  float texel_weight_rga[216];
  float texel_weight_rgb[216];

  float texel_weight_rg[216];
  float texel_weight_rb[216];
  float texel_weight_gb[216];
  float texel_weight_ra[216];

  float texel_weight_r[216];
  float texel_weight_g[216];
  float texel_weight_b[216];
  float texel_weight_a[216];

  int contains_zeroweight_texels;
} error_weight_block;

typedef struct {
  const char *kern_source;
  int extra_flags;
  size_t sortkey_width;

  int numCols;
  bool *nullsFirst;
} GpuSortPlan;

typedef struct {
  int n, lda, j0;
  short ipiv[32];
} claswp_params_t;

typedef struct _module_state_ {
  cl_program program;
  struct _module_state_ *next;
} module_state;

typedef struct {
  MotionSystem motionSystem;
} TriangleMotionMeshParam;

typedef struct {
  block_t block[25];
} macroblock_coeffs_t;

typedef struct krb5tgs_17 {
  unsigned int user[128];
  unsigned int domain[128];
  unsigned int account_info[512];
  unsigned int account_info_len;

  unsigned int checksum[3];
  unsigned int edata2[5120];
  unsigned int edata2_len;

} krb5tgs_17_t;

typedef struct {
  T_Block blk[((((1 << 15)) + (1) - 1) / (1))];
} T_HugeArray;

typedef struct {
  unsigned int length;
  uchar v[255];
} gpg_password;

typedef struct pbkdf2_sha256 {
  unsigned int salt_buf[16];

} pbkdf2_sha256_t;

typedef struct CvMoments {
  double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
  double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
  double inv_sqrt_m00;
} CvMoments;

typedef struct androidfde_tmp {
  unsigned int ipad[5];
  unsigned int opad[5];

  unsigned int dgst[10];
  unsigned int out[10];

} androidfde_tmp_t;

typedef struct krb5pa_18 {
  unsigned int user[128];
  unsigned int domain[128];
  unsigned int account_info[512];
  unsigned int account_info_len;

  unsigned int checksum[3];
  unsigned int enc_timestamp[32];
  unsigned int enc_timestamp_len;

} krb5pa_18_t;

typedef struct __attribute__((aligned(4))) {
  uchar numBxDFs;
  uchar hasBump;
  unsigned int idx_bump __attribute__((aligned(4)));
} MaterialInfo;

typedef struct {
  float r;
  float g;
  float b;
} RGB;

typedef struct {
  int rotlist_const[64];
} kernelconstant_rotlist;

typedef struct {
  float3 center;
  float3 lastPos;
  float3 position;
  float3 orientation;
  float3 velocity;
  float3 rotation;
  float radius;
  float radius2;
  float mass;
  float scale;
  RGB32 float3;
  int type;
  unsigned int index;
  unsigned int boolBits;
} ObjectInfo;

typedef struct {
  float a, b, c;
} index;

typedef struct {
  float *buf;
  int dim[1];
} __PSGrid1DFloatDev;

typedef struct {
  uint3 LocalWorkSize;
  uint3 TotalDimSize;
  IGIL_WalkerData WalkerArray[8];
} IGIL_WalkerEnumeration;

typedef enum __NhlCBTask { _NhlcbADD, _NhlcbDELETE, _NhlcbCALL } _NhlCBTask;

typedef union vconv32 {
  unsigned long long v32;

  struct {
    unsigned short a;
    unsigned short b;

  } v16;

  struct {
    unsigned char a;
    unsigned char b;
    unsigned char c;
    unsigned char d;

  } v8;

} vconv32_t;

typedef union {
  int value;
  short svalue[2];
} intShortUnion;

typedef union {
  struct {
    unsigned int a, b, c, d;
  };
  ulong res;
} tyche_state;

typedef union {
  int64_t v_int64;
  double v_float64;
  void *v_handle;
  const char *v_str;
} TVMValue;

typedef union {

} gpu_double_mem;

typedef union {
  unsigned int uint_val;
  float float_val;
} uintflt;

typedef union {
  unsigned int Int32;
  ulong Int64;
  float Float;
  double Double;
} llvmBitCastUnion;

typedef union union_type {
  float4 float4d;
  uint4 uint4d;
} typedef_union_type;

typedef union _Uint_and_Float {
  unsigned int uint_value;
  float float_value;
} Uint_and_Float;

typedef union FPtr {
  void *v;
  float *f;
  cl_double *d;
  cl_float2 *f2;
  cl_double2 *d2;
} FPtr;

typedef union {
  unsigned int u;
  int i;
} cb_t;

typedef union {
  unsigned int words[200 / sizeof(unsigned int)];
  uint2 uint2s[200 / sizeof(uint2)];
  uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;

typedef union TfLitePtrUnion {

  int *i32;
  int64_t *i64;
  float *f;
  TfLiteFloat16 *f16;
  char *raw;
  const char *raw_const;
  uint8_t *uint8;
  bool *b;
  int16_t *i16;
  TfLiteComplex64 *c64;
  signed char *int8;

  void *data;
} TfLitePtrUnion;

typedef union {
  unsigned int U4[16];
  ulong U8[8];
} U_HASH;

typedef union {
  uint4 m_int4;
  int m_ints[4];
} intconv;

typedef union u_primitive {
  t_plane plane;
  t_sphere sphere;
  t_cylinder cylinder;
  t_cone cone;
  t_torus torus;
  t_disk disk;
  t_rectangle rectangle;
  t_parallelogram parallelogram;
  t_triangle triangle;
  t_paraboloid paraboloid;
} t_primitive;

typedef union aes_block_u {
  ulong data[(16 / sizeof(unsigned long))];
  uchar bytes[16];
} aes_block_t;

typedef union FooUnion {
  int *ptrField;
} FooUnion;

typedef union LPtr {
  __local float *f;
  __local double *d;
  __local float2 *f2v;
  __local double2 *d2v;
  __local float4 *f4v;
  __local double4 *d4v;
  __local float8 *f8v;
  __local double8 *d8v;
  __local float16 *f16v;
  __local double16 *d16v;
} LPtr;

typedef union u_rgb {
  unsigned int c;
  unsigned char bgra[4];
} t_rgb;

typedef union {

} gpu_realw_mem;

typedef union {

} gpu_int_mem;

typedef union {
  int b1;
  float b2;
} transparent_u __attribute__((__transparent_union__));

typedef union {
  unsigned int h[32];
  ulong h2[16];
  uint4 h4[8];
  ulong4 h8[4];
} LyraState;

typedef union {
  ushort16 vectors;
  ushort varray[16];
} vectorUnion;

typedef union ArgMultiplier {
  float argFloat;
  cl_double argDouble;
  clFloatComplex argFloatComplex;
} ArgMultiplier;

typedef union {
  float v;
  float a[64];
} v_and_a;

typedef union {
  struct {
    float x;
    float y;
  };
  float s[2];
} RT_Point2f;

typedef union {
  unsigned int words[64 / sizeof(unsigned int)];
  uint2 uint2s[64 / sizeof(uint2)];
  uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
  unsigned int h4[8];
  ulong h8[4];
  uint4 h16[2];
  ulong2 hl16[2];
  ulong4 h32;
} hash2_t;

typedef union PPtr {
  float *f;
  double *d;
  float2 *f2v;
  double2 *d2v;
  float4 *f4v;
  double4 *d4v;
  float8 *f8v;
  double8 *d8v;
  float16 *f16v;
  double16 *d16v;
} PPtr;

typedef union {
  float value;
  unsigned int _short;
} float_shape_type;

typedef union {
  struct {
    cl_uint type;
    cl_uint data[5];
  } raw;
  struct {
    cl_uint type;
    cl_char unused[17];
    cl_char bus;
    cl_char device;
    cl_char function;
  } pcie;
} cl_device_topology_amd;

typedef union {
  ushort8 vec;
  ushort array[8];
} vec_array;

typedef union TF_TString_Union {
  TF_TString_Large large;
  TF_TString_Offset offset;
  TF_TString_View view;
  TF_TString_Raw raw;
} TF_TString_Union;
