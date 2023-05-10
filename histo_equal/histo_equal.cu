/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define N_IMAGES 10000

#define WARP_SIZE 32
#define NUM_WARPS 32
#define IMAGE_SIZE 65536
#define THREADS_PER_BLOCK 1024

typedef unsigned char uchar;

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
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
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
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;

    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    	int mapped = map[img_in[i]];
        img_out[i] = mapped;
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
    	// if(i < 10 * IMG_WIDTH * IMG_HEIGHT)
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GREY_SCALE 256

__device__ void arr_min(unsigned int arr[], int arr_size) {
    int threadId = threadIdx.x;
	int half_length = GREY_SCALE / 2;
	while(half_length >= 1){
        if(threadId < half_length){
            if(arr[threadId] == 0){
                arr[threadId] = arr[threadId + half_length];
            } else if(arr[threadId + half_length] && arr[threadId + half_length] < arr[threadId]){
                arr[threadId] = arr[threadId + half_length];
            }
        }
		__syncthreads();
		half_length/=2;
	}
}

__device__ void prefix_sum(unsigned int arr[], int arr_size) {
    int threadId = threadIdx.x;
	int increment;
	for(int stride =1; stride < GREY_SCALE; stride *=2){
		if(threadId >= stride && threadId < GREY_SCALE){
			increment = arr[threadId - stride];
		}
		__syncthreads();
		if(threadId >= stride && threadId < GREY_SCALE){
			arr[threadId]+= increment;
		}
		__syncthreads();
	}
	__syncthreads();
	//possible optimization: copy to the global mem paralell to working with shared
}

__device__ void gpu_histogram(uchar* img_in, unsigned int *hist){
    int threadId = threadIdx.x;
    int blockImageStart = blockIdx.x * IMAGE_SIZE;
    int MaxIndex = blockImageStart + IMAGE_SIZE - 1;
    for(int i = blockImageStart + threadId; i <= MaxIndex; i+=THREADS_PER_BLOCK){
        atomicAdd(&(hist[img_in[i]]), 1);
    }
	syncthreads();
}

//the corrected function I think should be:
// clarification: the previous implementation restricted to only one threadblock processing the whole image
// the commented new implemtation isn't restricted to any number of threadblocks and in the case 65536 / 1024 = 64,
//each thread in each block will process exactly one pixel (will add it to the histogram)
/* __device__ void gpu_histogram(uchar* img_in, unsigned int *hist){
    int threadId = threadIdx.x;
    int blockImageStart = blockIdx.x * THREADS_PER_BLOCK;
    int MaxIndex = blockImageStart + THREADS_PER_BLOCK - 1;
    for(int i = blockImageStart + threadId; i <= MaxIndex; i+=THREADS_PER_BLOCK){
        atomicAdd(&(hist[img_in[i]]), 1);
    }
    syncthreads();
}
*/

__device__ void map(uchar *mapped, unsigned int *cdf, unsigned int cdf_min){
    int threadId = threadIdx.x;
    if(threadId < GREY_SCALE){
        int map_value = (float)(cdf[threadId] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        mapped[threadId] = (uchar)map_value;
    }
	__syncthreads();
}

__device__ void update_image(uchar *img_in, uchar *mapped, uchar *img_out){
	    int threadId = threadIdx.x;
	    int blockImageStart = blockIdx.x * IMAGE_SIZE;
	    int MaxIndex = blockImageStart + IMAGE_SIZE - 1;
	    for(int i = blockImageStart + threadId; i <= MaxIndex; i+=THREADS_PER_BLOCK){
	    	img_out[i] = mapped[img_in[i]];
	    }
		syncthreads();
}

__global__ void process_image_kernel(uchar *in, uchar *out) {
	__shared__ unsigned int hist[GREY_SCALE];
	__shared__ unsigned int a_min[GREY_SCALE];
    int threadId  = threadIdx.x;
    if(threadId < GREY_SCALE){
        hist[threadId] = 0;
    }
	__syncthreads(); // Ipus hist
	gpu_histogram(in, hist);
	prefix_sum(hist, GREY_SCALE);
    if(threadId < GREY_SCALE){
        a_min[threadId] = hist[threadId];
    }
    __syncthreads();
	arr_min(a_min, GREY_SCALE);
	__shared__ uchar mapped[GREY_SCALE];
	map(mapped, hist, a_min[0]);
	update_image(in, mapped, out);
    return ;
}


int main() {
///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0));
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change
    //allocate GPU memory for a single input image and a single output image
    uchar *my_img_in, *my_img_out;
    cudaMalloc(&my_img_in, IMG_HEIGHT*IMG_WIDTH*sizeof(uchar));
    cudaMalloc(&my_img_out, IMG_HEIGHT*IMG_WIDTH*sizeof(uchar));
    t_start = get_time_msec(); //Do not change
    for(int i = 0; i < N_IMAGES; ++i){
        //1. copy the relevant image from images_in to the GPU memory you allocated
        cudaMemcpy(my_img_in, &(images_in[IMG_WIDTH * IMG_HEIGHT*i]),
        		   IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
        //2. invoke GPU kernel on this image
       process_image_kernel<<<1, 1024>>>(my_img_in, my_img_out);
       cudaDeviceSynchronize();
       cudaError_t error=cudaGetLastError();
       if (error!=cudaSuccess) {
    	   fprintf(stderr,"Kernel execution failed:%s\n",cudaGetErrorString(error));
    	   cudaFree(my_img_in);
    	   cudaFree(my_img_out);
    	   return 1;
       }
       //3. copy output from GPU memory to relevant location in images_out_gpu_serial
        cudaMemcpy(&(images_out_gpu_serial[IMG_WIDTH * IMG_HEIGHT*i]), my_img_out,
        		   IMG_WIDTH * IMG_HEIGHT * sizeof(uchar) , cudaMemcpyDeviceToHost);
    }
    cudaFree(my_img_in);
    cudaFree(my_img_out);
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change


    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
    uchar *my_img_in4, *my_img_out4;
    int imagesMemorySize = IMG_HEIGHT * IMG_WIDTH * sizeof(uchar) * N_IMAGES;
    cudaMalloc(&(my_img_in4), imagesMemorySize);
    cudaMalloc(&(my_img_out4), imagesMemorySize);

    t_start = get_time_msec(); //Do not change

    //TODO: copy all input images from images_in to the GPU memory you allocated
    cudaMemcpy(my_img_in4,  images_in, imagesMemorySize, cudaMemcpyHostToDevice);
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    process_image_kernel<<<N_IMAGES, 1024>>>(my_img_in4, my_img_out4);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"Kernel execution failed:%s\n",cudaGetErrorString(error));
        //TODO: free memory
        return 1;
    }
    //TODO: copy output images from GPU memory to images_out_gpu_bulk
    cudaMemcpy(images_out_gpu_bulk, my_img_out4, imagesMemorySize, cudaMemcpyDeviceToHost);

    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange


    return 0;
}
