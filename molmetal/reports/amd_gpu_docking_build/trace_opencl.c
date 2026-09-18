/* Independent smoke instrumentation: observe completed real GPU kernels.
 * This serializes kernel launches, so these times are not benchmark results.
 */
#define CL_TARGET_OPENCL_VERSION 200
#define _GNU_SOURCE
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <time.h>

typedef cl_int (*enqueue_fn)(cl_command_queue, cl_kernel, cl_uint,
    const size_t *, const size_t *, const size_t *, cl_uint,
    const cl_event *, cl_event *);

static double seconds(struct timespec a, struct timespec b) {
    return (b.tv_sec-a.tv_sec) + (b.tv_nsec-a.tv_nsec)*1e-9;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel,
    cl_uint dim, const size_t *offset, const size_t *global,
    const size_t *local, cl_uint nwait, const cl_event *waits, cl_event *out) {
    enqueue_fn original = (enqueue_fn)dlsym(RTLD_NEXT, "clEnqueueNDRangeKernel");
    if (!original) return CL_INVALID_OPERATION;
    cl_event own_event = NULL;
    cl_event *result_event = out ? out : &own_event;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cl_int result = original(queue, kernel, dim, offset, global, local, nwait, waits, result_event);
    if (result == CL_SUCCESS) {
        cl_int wait_result = clWaitForEvents(1, result_event);
        clock_gettime(CLOCK_MONOTONIC, &end);
        cl_device_id device;
        char device_name[256] = {0}, name[256] = {0};
        cl_int status = -999;
        cl_ulong profile_start = 0, profile_end = 0;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, NULL);
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(name), name, NULL);
        clGetEventInfo(*result_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
        cl_int p0 = clGetEventProfilingInfo(*result_event, CL_PROFILING_COMMAND_START, sizeof(profile_start), &profile_start, NULL);
        cl_int p1 = clGetEventProfilingInfo(*result_event, CL_PROFILING_COMMAND_END, sizeof(profile_end), &profile_end, NULL);
        fprintf(stderr, "OPENCL_TRACE kernel=%s device=%s enqueue=%d wait=%d status=%d host_wait_s=%.9f profile_start=%llu profile_end=%llu profile_valid=%d dims=%u global=%zu,%zu,%zu\n",
            name, device_name, result, wait_result, status, seconds(start,end),
            (unsigned long long)profile_start, (unsigned long long)profile_end,
            p0 == CL_SUCCESS && p1 == CL_SUCCESS && profile_end >= profile_start,
            dim, global[0], dim > 1 ? global[1] : 1, dim > 2 ? global[2] : 1);
        if (!out) clReleaseEvent(own_event);
    }
    return result;
}
