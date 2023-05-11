/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_DIMENSION 32
#define NREQUESTS 10000

#define SINGLE_QUEUE_SIZE 10

typedef unsigned char uchar;
/////////////     GPU     /////////////////
__global__ void test_kernel(volatile uchar* cpu_gpu_queue, volatile int* cpu_gpu_flags, 
    volatile uchar* gpu_cpu_queue, volatile int* gpu_cpu_flags, volatile int* running);
__device__ void gpu_process_image_device(uchar *in, uchar *out);
/////////////     CPU     /////////////////
int blocksPerMP(int major, int minor);
int max_thread_blocks(int threads_num);
int max_thread_blocks(int threads_num);

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

double distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    double distance_sqr = 0;
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we won't load actual files. just fill the images with random bytes */
void load_images(uchar *images) {
    srand(0);
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        images[i] = rand() % 256;
    }
}

__device__ int arr_min(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}


enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {
    // int end_counter = 0;

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images_in; /* we concatenate all  one huge array */
    uchar *images_out;
    CUDA_CHECK( cudaHostAlloc(&images_in, NREQUESTS * SQR(IMG_DIMENSION), 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    load_images(images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&images_in[img_idx * SQR(IMG_DIMENSION)], &images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    double total_distance = 0;

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");

    uchar *images_out_from_gpu;
    CUDA_CHECK( cudaHostAlloc(&images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    do {
        uchar *gpu_image_in, *gpu_image_out;
        CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

        t_start = get_time_msec();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            CUDA_CHECK(cudaMemcpy(gpu_image_in, &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
            gpu_process_image<<<1, 1024>>>(gpu_image_in, gpu_image_out);
            CUDA_CHECK(cudaMemcpy(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)], gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
        }
        total_distance += distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("distance from baseline %lf (should be zero)\n", total_distance);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    /* TODO allocate / initialize memory, streams, etc... */
    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));
    
     int streams_queue[64];
    for (int i = 0; i < 64; i++) {
        streams_queue[i] = -1;
    }
    int streams_counters[64] = { 0 };

    cudaStream_t streams[64]; //added
    for (int i = 0; i < 64; i++) { //added
        CUDA_CHECK(cudaStreamCreate(&streams[i])); //added
    } //added

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
     	int count = 0;
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {

            /* TODO query (don't block) streams for any completed requests.
             * update req_t_end of completed requests
             */
            int avail_idx = -1;
            while (avail_idx == -1) {
                for (int i = 0; i < 64; i++) {
                    cudaError_t err_start = cudaStreamQuery(streams[i]);
                    if (err_start != cudaErrorNotReady && avail_idx == -1) {
                        avail_idx = i;

                    }
                    if (err_start != cudaErrorNotReady
                            && streams_queue[i] != -1) {
                        if (req_t_start[streams_queue[i]] != 0) {
                        	if(req_t_end[streams_queue[i]] == 0){
                        		req_t_end[streams_queue[i]] = get_time_msec();
                        		count++;
                        		streams_queue[i] = -1;
                        	}
                        }
                    }
                }
            }
            streams_counters[avail_idx]++;
            streams_queue[avail_idx] = img_idx;

            if (!rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }

            req_t_start[img_idx] = get_time_msec();

            uchar *gpu_image_in, *gpu_image_out;
            CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
            CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

            /* TODO place memcpy's and kernels in a stream */
	    // the following if seems like a bug because
	    // for img_idx >= 64, the corresponding memory transfers and kernel launches
	    // will not be asynchronous and will be executed sequentially on the default stream
            if (img_idx < 64) {
                streams_queue[avail_idx] = img_idx;
            }
            cudaMemcpyAsync(gpu_image_in,
                    &images_in[img_idx * SQR(IMG_DIMENSION)],
                    SQR(IMG_DIMENSION), cudaMemcpyHostToDevice,
                    streams[avail_idx]);
            gpu_process_image<<<1, 1024, 0, streams[avail_idx]>>>(gpu_image_in,
                    gpu_image_out);
            cudaMemcpyAsync(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)],
                    gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost,
                    streams[avail_idx]);
        }

        while(count < NREQUESTS){
        	for (int i = 0; i < 64; i++) {
        		if (streams_queue[i] != -1 && req_t_start[streams_queue[i]] != 0) {
        			if(req_t_end[streams_queue[i]] == 0){
        				req_t_end[streams_queue[i]] = get_time_msec();
        				count++;
        			}
        		}
        	}
        }
        
        /* TODO now make sure to wait for all streams to finish */
        CUDA_CHECK(cudaDeviceSynchronize());

    } else if (mode == PROGRAM_MODE_QUEUE) {
        int num_blocks = max_thread_blocks(threads_queue_mode);
        
        size_t queue_size_bytes = num_blocks * SINGLE_QUEUE_SIZE * SQR(IMG_DIMENSION) * sizeof(uchar);
        size_t flags_size_bytes = num_blocks * SINGLE_QUEUE_SIZE * sizeof(int);
        volatile int *running;
        CUDA_CHECK(cudaHostAlloc(&running, sizeof(int), cudaHostAllocMapped));

        volatile uchar *cpu_gpu_queue, *gpu_cpu_queue;
        volatile int *cpu_gpu_flags, *gpu_cpu_flags;
        
        int count = 0;

        CUDA_CHECK(cudaHostAlloc(&cpu_gpu_queue, queue_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&gpu_cpu_queue, queue_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&cpu_gpu_flags, flags_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&gpu_cpu_flags, flags_size_bytes, cudaHostAllocMapped));

        for(int i=0; i<num_blocks * SINGLE_QUEUE_SIZE; i++){
            cpu_gpu_flags[i] = -1;
            gpu_cpu_flags[i] = -1;
        }
        *running = 0;
        __sync_synchronize();
        // TODO launch GPU consumer-producer kernel
        test_kernel<<<num_blocks, threads_queue_mode>>>(cpu_gpu_queue, 
            cpu_gpu_flags, gpu_cpu_queue, gpu_cpu_flags, running);
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            /* TODO check producer consumer queue for any responses.
             * don't block. if no responses are there we'll check again in the next iteration
             * update req_t_end of completed requests 
             */
            for(int i=0; i<num_blocks * SINGLE_QUEUE_SIZE; i++){
                if(gpu_cpu_flags[i] != -1){
                    int out_image_index = gpu_cpu_flags[i];
                    for(int k=0; k<SQR(IMG_DIMENSION); k++){
                        images_out_from_gpu[out_image_index * SQR(IMG_DIMENSION) + k] 
                            = gpu_cpu_queue[i * SQR(IMG_DIMENSION) + k];
                    }
                    __sync_synchronize();
                    gpu_cpu_flags[i] = -1;
                    __sync_synchronize();
                    req_t_end[out_image_index] = get_time_msec();
                    count++;
                }
            }
            __sync_synchronize();

            if (!rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }

            req_t_start[img_idx] = get_time_msec();

            /* TODO push task to queue */
            bool done = false;
            while(!done){
                for(int i=0; i<num_blocks * SINGLE_QUEUE_SIZE; i++){
                    if(cpu_gpu_flags[i] == -1){
                        for(int k=0; k<SQR(IMG_DIMENSION); k++){
                            cpu_gpu_queue[i * SQR(IMG_DIMENSION) + k] = 
                                images_in[img_idx * SQR(IMG_DIMENSION) + k];
                        }
                        __sync_synchronize();
                        cpu_gpu_flags[i] = img_idx;
                        __sync_synchronize();
                        done = true;
                        break;
                    }
                }
            }
            __sync_synchronize();
        }
        /* TODO wait until you have responses for all requests */
        while(count < NREQUESTS){
                for(int i=0; i<num_blocks * SINGLE_QUEUE_SIZE; i++){
                if(gpu_cpu_flags[i] != -1){
                    int out_image_index = gpu_cpu_flags[i];
                    for(int k=0; k<SQR(IMG_DIMENSION); k++){
                        images_out_from_gpu[out_image_index * SQR(IMG_DIMENSION) + k] 
                            = gpu_cpu_queue[i * SQR(IMG_DIMENSION) + k];
                    }
                    __sync_synchronize();
                    gpu_cpu_flags[i] = -1;
                    __sync_synchronize();
                    req_t_end[out_image_index] = get_time_msec();
                    count++;
                }
            }        
        }
        while(*running<num_blocks);
        *running = num_blocks+1;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFreeHost((int*)running));
        CUDA_CHECK(cudaFreeHost((uchar*)cpu_gpu_queue));
        CUDA_CHECK(cudaFreeHost((uchar*)gpu_cpu_queue));
        CUDA_CHECK(cudaFreeHost((int*)cpu_gpu_flags));
        CUDA_CHECK(cudaFreeHost((int*)gpu_cpu_flags));
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    total_distance = distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
    double avg_latency = 0;
    
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}

struct message_t{
    uchar* data;
    int id;
};

struct queue_t{
    uchar* data[SINGLE_QUEUE_SIZE];
    int index[SINGLE_QUEUE_SIZE];
};

/////////////     GPU     /////////////////
__device__ void dequeue_request(volatile uchar* queue, volatile int* flags, 
    uchar* image_out, int* image_id){
    int tid = threadIdx.x;
    __shared__ int index;
    if(tid==0){
    index = -1;
        for(int i=0; i<SINGLE_QUEUE_SIZE; i++){
            if(flags[i] != -1){
                index = i;
                *image_id = flags[i];
                break;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if(index != -1){
      for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x){
              image_out[i] = queue[index*SQR(IMG_DIMENSION)+i];
      }
    }
    __syncthreads();
    __threadfence_system();
    if(tid==0 && index != -1){
        flags[index] = -1;   
    }
    __threadfence_system();
}

__device__ void enqueue_response(volatile uchar* queue, volatile int* flags, 
    uchar* image_out, int image_id){
    int tid = threadIdx.x;
    __shared__ int index;
    if(tid==0){
    index = -1;
        for(int i=0; index==-1; i=(i+1)%SINGLE_QUEUE_SIZE){
            if(flags[i] == -1){
                index = i;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if(index != -1){
      for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x){
               queue[index*SQR(IMG_DIMENSION)+i] = image_out[i] ;
      }
    }
    __syncthreads();
    __threadfence_system();
    if(tid==0 && index != -1){
        flags[index] = image_id;   
    }
    __syncthreads();
    __threadfence_system();
}

__global__ void test_kernel(volatile uchar* cpu_gpu_queue, volatile int* cpu_gpu_flags, 
    volatile uchar* gpu_cpu_queue, volatile int* gpu_cpu_flags, volatile int* running){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    __shared__ uchar image_in[SQR(IMG_DIMENSION)];
    __shared__ uchar image_out[SQR(IMG_DIMENSION)];
    __shared__ int image_id;
    __shared__ int queue_index;
    __shared__ int flags_index;
    __shared__ bool started;
    if(tid==0){
        started = false;
        queue_index = bid * SINGLE_QUEUE_SIZE * SQR(IMG_DIMENSION);
        flags_index = bid * SINGLE_QUEUE_SIZE;
    }
    __syncthreads();
    __threadfence_system();
    while(*running < num_blocks+1){
        if(tid==0){
            if(!started){
                started = true;
                atomicAdd((int*)running, 1);
            }
            image_id = -1;
        }
        __syncthreads();
        __threadfence_system();
        dequeue_request(cpu_gpu_queue + queue_index, 
            cpu_gpu_flags + flags_index, (uchar*)image_in, &image_id);
        __syncthreads();
        __threadfence_system();
        if(image_id != -1){
            gpu_process_image_device((uchar*)image_in, (uchar*)image_out);
        }
        __syncthreads();
        __threadfence_system();
        if(image_id != -1){
            enqueue_response(gpu_cpu_queue + queue_index, gpu_cpu_flags + flags_index, 
                (uchar*)image_out, image_id);
        }
        __syncthreads();
        __threadfence_system();
    }
}

__device__ void gpu_process_image_device(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

/////////////     CPU     /////////////////
int blocksPerMP(int major, int minor){
    switch(major) {
      case 1 :
        return 8;
      case 2 :
        return 8;
      case 3 :
        return 16;
      case 5 :
        return 32;
      case 6 :
        return 32;
      case 7 :
        if(minor < 5){
          return 32;
        } else {
          return 16;
        }
      default :
        return -1;
    }
}


int max_thread_blocks(int threads_num){
	struct cudaDeviceProp devProp;
	CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
	int regs_per_thread = 32;
	int threads_per_threadblock = threads_num;
	int shared_mem_per_threadblock = sizeof(uchar)*SINGLE_QUEUE_SIZE*SQR(IMG_DIMENSION);
  int bound1 = devProp.sharedMemPerMultiprocessor/shared_mem_per_threadblock;
  int bound2 =  devProp.sharedMemPerMultiprocessor/shared_mem_per_threadblock;
  int bound3 = devProp.regsPerMultiprocessor/regs_per_thread/threads_per_threadblock;
  int tmp = bound1 < bound2 : bound1 : bound2;
  int min = tmp < bound3 : tmp : bound3;
	int max_threadblocks = devProp.multiProcessorCount * min;
  return max_threadblocks;
}
