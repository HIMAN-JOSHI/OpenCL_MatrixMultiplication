// This program demonstrates matrix multiplication using OpenCL.

// header files

// standard headers
#include<stdio.h>
#include<math.h> // fabs();
#include<stdlib.h> // exit();

// cuda headers
#include<CL/opencl.h>

// macros
#define BLOCK_WIDTH 64

// global variables
cl_platform_id oclPlatformId;
cl_device_id oclDeviceId;
cl_context oclContext;
cl_command_queue oclCommandQueue;
cl_program oclProgram;
cl_kernel oclKernel;

int* hostA = NULL;
int* hostB = NULL;
int* hostC = NULL;
int* gold = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

// OpenCL kernel
const char* oclSourceCode =
"__kernel void matrixMultiplicationGPU(__global int *A, __global int *B, __global int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)" \
"{" \
"int row = get_global_id(0);" \
"int column = get_global_id(1);" \
"if((row < numARows) && (column < numBColumns))" \
"{" \
"int value = 0;" \
"for(int k = 0; k < numAColumns; k++)"
"{" \
"int a=A[row * numAColumns + k];" \
"int b=B[k * numBColumns + column];" \
"value += a*b;" \
"}" \
"C[row * numCColumns + column]=value;" \
"}" \
"}";

int main(int argc, char* argv[]) {

	// function declarations
	void InitA(int* data, int, int);
	void InitB(int* data, int, int);
	void matrixMultiplicationCPU(int*, int*, int*, int, int, int, int);
	void cleanup(void);

	// variable declarations
	int numARows = BLOCK_WIDTH;
	int numAColumns = BLOCK_WIDTH;
	int numBRows = BLOCK_WIDTH;
	int numBColumns = BLOCK_WIDTH;
	int numCRows = numARows;
	int numCColumns = numBColumns;

	int numGoldRows = numARows;
	int numGoldColumns = numBColumns;

	int sizeA = numARows * numAColumns * sizeof(int);
	int sizeB = numBRows * numBColumns * sizeof(int);
	int sizeC = numCRows * numCColumns * sizeof(int);
	int sizeGold = numGoldRows * numGoldColumns * sizeof(int);

	cl_int result;

	// code
	// host memory allocation
	hostA = (int*)malloc(sizeA);
	if (hostA == NULL) {
		printf("Host memory allocation failed for hostA matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostB = (int*)malloc(sizeB);
	if (hostB == NULL) {
		printf("Host memory allocation failed for hostB matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostC = (int*)malloc(sizeC);
	if (hostC == NULL) {
		printf("Host memory allocation failed for hostC matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (int*)malloc(sizeGold);
	if (gold == NULL) {
		printf("Host memory allocation failed for 'gold' matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// printing matrix dimensions and sizes
	printf("The Dimensions of Matrix 'hostA' are : %d x %d \n", numARows, numAColumns);
	printf("The Dimensions of Matrix 'hostB' are : %d x %d \n", numBRows, numBColumns);
	printf("The Dimensions of Matrix 'hostC' are : %d x %d \n", numCRows, numCColumns);
	printf("The Dimensions of Matrix 'gold' are : %d x %d \n", numGoldRows, numGoldColumns);

	// fill source matrices
	InitA(hostA, numARows, numAColumns);
	InitB(hostB, numBRows, numBColumns);

	// 1. Get platform's Id
	result = clGetPlatformIDs(1, &oclPlatformId, NULL);
	if (result != CL_SUCCESS) {
		printf("clGetPlatformIDs() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 2. Get OpenCL supporting GPU device's ID
	result = clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, 1, &oclDeviceId, NULL);
	if (result != CL_SUCCESS) {
		printf("clGetDeviceIDs() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 3. Create OpenCL compute context
	oclContext = clCreateContext(NULL, 1, &oclDeviceId, NULL, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateContext() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 4. Create command queue
	oclCommandQueue = clCreateCommandQueueWithProperties(oclContext, oclDeviceId, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateCommandQueue() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 5. Create OpenCL program from .cl
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&oclSourceCode, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateProgramwithSource() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 6. Build the OpenCL program - compile and link a program executable from the program source or binary.
	result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (result != CL_SUCCESS) {

		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(oclProgram, oclDeviceId, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &len);
		printf("Program build log: %s\n", buffer);
		printf("clBuildProgram() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 7. Create OpenCL kernel by function name.
	oclKernel = clCreateKernel(oclProgram, "matrixMultiplicationGPU", &result);
	if (result != CL_SUCCESS) {
		printf("clCreateKernel() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 8. Device memory allocation
	deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeA, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateBuffer() failed for 1st input matrix: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeB, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateBuffer() failed for 2nd input matrix: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);

	}

	deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeC, NULL, &result);
	if (result != CL_SUCCESS) {
		printf("clCreateBuffer() failed for 3rd input matrix: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 10. Set 0-based 0th argument i.e. deviceA
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceA);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 1st argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Set 0-based 1st argument i.e. deviceB
	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceB);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 2nd argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Set 0-based 2nd argument i.e. deviceC
	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceC);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 3rd argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Set 0-based 3rd argument i.e. numARows
	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&numARows);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 4th argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Set 0-based 4th argument i.e. numAColumns
	result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&numAColumns);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 5th argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// set 0-based 5th argument i.e. numBColumns
	result = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void*)&numBColumns);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 6th argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// set 0-based 6th argument i.e. numCColumns
	result = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void*)&numCColumns);
	if (result != CL_SUCCESS) {
		printf("clSetKernelArg() failed for 6th argument: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 10. Write above 'input' device buffer into device memory
	result = clEnqueueWriteBuffer(oclCommandQueue,
		deviceA,
		CL_FALSE,
		0,
		sizeA,
		hostA,
		0,
		NULL,
		NULL);
	if (result != CL_SUCCESS) {
		printf("clEnqueueWriteBuffer() failed for 1st input device buffer: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);
	if (result != CL_SUCCESS) {
		printf("clEnqueueWriteBuffer() failed for 2nd input device buffer: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 11. kernel configuration
	size_t globalWorkSize[2];
	globalWorkSize[0] = BLOCK_WIDTH;
	globalWorkSize[1] = BLOCK_WIDTH;

	/*  Enqueues a command to execute a kernel on a device.

		'ND' stands for N-Dimensions
	*/
	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel,
		2, // 2-Dimensions
		NULL, // Reserved param
		globalWorkSize,
		NULL,
		0,
		NULL,
		NULL);
	if (result != CL_SUCCESS) {
		printf("clEnqueueNDRangeKernel() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// 12. Finish the OpenCL command-queue (i.e. allow OpenCL to run all
	// the commands until this point in the command queue.)
	clFinish(oclCommandQueue);

	// 13. Read back result from the device (i.e. from deviceOutput) into
	// CPU variable (i.e. hostOutput)
	result = clEnqueueReadBuffer(oclCommandQueue, deviceC,
		CL_TRUE, // block the read operations
		0, // start reading from 0th offset.
		sizeC, // size of data to be read.
		hostC, // buffer where data is to be read into.
		0, // no events in wait list so 0.
		NULL, NULL);
	if (result != CL_SUCCESS) {
		printf("clEnqueueReadBuffer() failed: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	// matrix multiplication on the host (i.e. CPU)
	matrixMultiplicationCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns);

	// comparison
	int breakValue = -1;
	bool bAccuracy = true;
	for (int i = 0; i < numCRows * numCColumns; i++) {

		int val1 = gold[i];
		int val2 = hostC[i];
		if (val1 != val2) {
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	char str[128];
	if (bAccuracy == false) {
		sprintf_s(str, "Comparison of CPU and GPU Matrix Multiplication is not accurate at array index %d", breakValue);
	}
	else {
		sprintf_s(str, "Comparison of CPU and GPU Matrix Multiplication is accurate.");
	}

	printf("%s\n", str);

	cleanup();

	return (0);

}

void InitA(int* data, int row, int col) {

	int num = 1;

	// code
	for (int i = 0; i < row; i++) {

		for (int j = 0; j < col; j++) {
			*(data + i * col + j) = num;
			num++;
		}
	}
}

void InitB(int* data, int row, int col) {

	int num = BLOCK_WIDTH;

	// code
	for (int i = 0; i < row; i++) {

		for (int j = 0; j < col; j++) {

			*(data + i * col + j) = num;
			num--;
		}
	}
}

void matrixMultiplicationCPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumns) {

	// code
	for (int i = 0; i < numARows; ++i) {
		for (int j = 0; j < numBColumns; ++j) {

			int value = 0.0f;
			for (int k = 0; k < numAColumns; ++k) {
				int a = A[i * numAColumns + k];
				int b = B[k * numBColumns + j];
				value += a * b;
			}
			C[i * numCColumns + j] = value;
		}
	}
}

void cleanup() {

	// code
	if (deviceC) {
		clReleaseMemObject(deviceC);
		deviceC = NULL;
	}

	if (deviceB) {
		clReleaseMemObject(deviceB);
		deviceB = NULL;
	}

	if (deviceA) {
		clReleaseMemObject(deviceA);
		deviceA = NULL;
	}

	if (oclKernel) {
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}

	if (oclProgram) {
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if (oclCommandQueue) {
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if (oclContext) {
		clReleaseContext(oclContext);
		oclContext = NULL;
	}

	if (gold) {
		free(gold);
		gold = NULL;
	}

	if (hostC) {
		free(hostC);
		hostC = NULL;
	}

	if (hostB) {
		free(hostB);
		hostB = NULL;
	}

	if (hostA) {
		free(hostA);
		hostA = NULL;
	}
}