/*
 * Copyright (C) 2002-2022 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

#include "osu_util_mpi.h"

MPI_Request request[MAX_REQ_NUM];
MPI_Status  reqstat[MAX_REQ_NUM];
MPI_Request send_request[MAX_REQ_NUM];
MPI_Request recv_request[MAX_REQ_NUM];

MPI_Aint disp_remote;
MPI_Aint disp_local;

/* A is the A in DAXPY for the Compute Kernel */
#define A 2.0
#define DEBUG 0
/*
 * We are using a 2-D matrix to perform dummy
 * computation in non-blocking collective benchmarks
 */
#define DIM 25
static float **a, *x, *y;

/* Validation multiplier constants*/
#define FLOAT_VALIDATION_MULTIPLIER 2.0
#define CHAR_VALIDATION_MULTIPLIER 7
#define CHAR_RANGE (int) pow(2, __CHAR_BIT__)

#ifdef _ENABLE_CUDA_
CUcontext cuContext;
#endif

char const *win_info[20] = {
    "MPI_Win_create",
#if MPI_VERSION >= 3
    "MPI_Win_allocate",
    "MPI_Win_create_dynamic",
#endif
};

char const *sync_info[20] = {
    "MPI_Win_lock/unlock",
    "MPI_Win_post/start/complete/wait",
    "MPI_Win_fence",
#if MPI_VERSION >= 3
    "MPI_Win_flush",
    "MPI_Win_flush_local",
    "MPI_Win_lock_all/unlock_all",
#endif
};

#ifdef _ENABLE_CUDA_KERNEL_
/* Using new stream for kernels on gpu */
static cudaStream_t stream;
/* Using new stream and events for UM buffer handling */
static cudaStream_t um_stream;
static cudaEvent_t start, stop;

static int is_alloc = 0;

/* Arrays on device for dummy compute */
static float *d_x, *d_y;
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

void set_device_memory (void * ptr, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)ptr;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaMemset(ptr, data, size));
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
#pragma acc parallel copyin(size) deviceptr(p)
            for (i = 0; i < size; i++) {
                p[i] = data;
            }
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
            ROCM_CHECK(hipMemset(ptr, data, size));
            break;
#endif
        default:
            break;
    }
}

int free_device_buffer (void * buf)
{
    if (buf == NULL)
        return 0;
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaFree(buf));
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            acc_free(buf);
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
            ROCM_CHECK(hipFree(buf));
            break;
#endif
        default:
            /* unknown device */
            return 1;
    }

    buf = NULL;
    return 0;
}

void *align_buffer (void * ptr, unsigned long align_size)
{
    unsigned long buf = (((unsigned long)ptr + (align_size - 1)) / align_size * align_size);
    return (void *) buf;
}


void usage_one_sided (char const * name)
{
    if (accel_enabled) {
        fprintf(stdout, "Usage: %s [options] [SRC DST]\n\n", name);
        fprintf(stdout, "SRC and DST are buffer types for the source and destination\n");
        fprintf(stdout, "SRC and DST may be `D' or `H' which specifies whether\n"
                        "the buffer is allocated on the accelerator device or host\n"
                        "memory for each mpi rank\n\n");
    } else {
        fprintf(stdout, "Usage: %s [options]\n", name);
    }

    fprintf(stdout, "Options:\n");

    if (accel_enabled) {
        fprintf(stdout, "  -d --accelerator <type>       accelerator device buffers can be of <type> "
                   "`cuda', `openacc', or `rocm'\n");
    }
    fprintf(stdout, "\n");

#if MPI_VERSION >= 3
    fprintf(stdout, "  -w --win-option <win_option>\n");
    fprintf(stdout, "            <win_option> can be any of the follows:\n");
    fprintf(stdout, "            create            use MPI_Win_create to create an MPI Window object\n");
    if (accel_enabled) {
        fprintf(stdout, "            allocate          use MPI_Win_allocate to create an MPI Window object (not valid when using device memory)\n");
    } else {
        fprintf(stdout, "            allocate          use MPI_Win_allocate to create an MPI Window object\n");
    }
    fprintf(stdout, "            dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    fprintf(stdout, "\n");
#endif

    fprintf(stdout, "  -s, --sync-option <sync_option>\n");
    fprintf(stdout, "            <sync_option> can be any of the follows:\n");
    fprintf(stdout, "            pscw              use Post/Start/Complete/Wait synchronization calls \n");
    fprintf(stdout, "            fence             use MPI_Win_fence synchronization call\n");
    if (options.synctype == ALL_SYNC) {
        fprintf(stdout, "            lock              use MPI_Win_lock/unlock synchronizations calls\n");
#if MPI_VERSION >= 3
        fprintf(stdout, "            flush             use MPI_Win_flush synchronization call\n");
        fprintf(stdout, "            flush_local       use MPI_Win_flush_local synchronization call\n");
        fprintf(stdout, "            lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
#endif
    }
    fprintf(stdout, "\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }
    fprintf(stdout, "  -x, --warmup ITER           number of warmup iterations to skip before timing"
                   "(default 100)\n");

    if (options.subtype == BW) {
        fprintf(stdout, "  -W, --window-size SIZE      set number of messages to send before synchronization (default 64)\n");
    }

    fprintf(stdout, "  -G, --graph tty,png,pdf    graph output of per"
            " iteration values.\n");
#ifdef _ENABLE_PAPI_
    fprintf(stdout, "  -P, --papi [EVENTS]:[PATH]     Enable PAPI support\n");
    fprintf(stdout, "                                 [EVENTS]       //Comma seperated list of PAPI events\n");
    fprintf(stdout, "                                 [PATH]         //PAPI output file path\n");
#endif
    fprintf(stdout, "  -i, --iterations ITER       number of iterations for timing (default 10000)\n");

    fprintf(stdout, "  -h, --help                  print this help message\n");
    fflush(stdout);
}

int process_one_sided_options (int opt, char *arg)
{
    switch(opt) {
        case 'w':
#if MPI_VERSION >= 3
            if (0 == strcasecmp(arg, "create")) {
                options.win = WIN_CREATE;
            } else if (0 == strcasecmp(arg, "allocate")) {
                options.win = WIN_ALLOCATE;
            } else if (0 == strcasecmp(arg, "dynamic")) {
                options.win = WIN_DYNAMIC;
            } else
#endif
            {
                return PO_BAD_USAGE;
            }
            break;
        case 's':
            if (0 == strcasecmp(arg, "pscw")) {
                options.sync = PSCW;
            } else if (0 == strcasecmp(arg, "fence")) {
                options.sync = FENCE;
            } else if (options.synctype== ALL_SYNC) {
                if (0 == strcasecmp(arg, "lock")) {
                    options.sync = LOCK;
                }
#if MPI_VERSION >= 3
                else if (0 == strcasecmp(arg, "flush")) {
                    options.sync = FLUSH;
                } else if (0 == strcasecmp(arg, "flush_local")) {
                    options.sync = FLUSH_LOCAL;
                } else if (0 == strcasecmp(arg, "lock_all")) {
                    options.sync = LOCK_ALL;
                }
#endif
                else {
                    return PO_BAD_USAGE;
                }
            } else {
                return PO_BAD_USAGE;
            }
            break;
        default:
            return PO_BAD_USAGE;
    }

    return PO_OKAY;
}

void usage_mbw_mr()
{
    if (accel_enabled) {
        fprintf(stdout, "Usage: osu_mbw_mr [options] [SRC DST]\n\n");
        fprintf(stdout, "SRC and DST are buffer types for the source and destination\n");
        fprintf(stdout, "SRC and DST may be `D', `H', or 'M' which specifies whether\n"
                        "the buffer is allocated on the accelerator device memory, host\n"
                        "memory or using CUDA Unified memory respectively for each mpi rank\n\n");
    } else {
        fprintf(stdout, "Usage: osu_mbw_mr [options]\n");
    }

    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -R=<0,1>, --print-rate         Print uni-directional message rate (default 1)\n");
    fprintf(stdout, "  -p=<pairs>, --num-pairs        Number of pairs involved (default np / 2)\n");
    fprintf(stdout, "  -W=<window>, --window-size     Number of messages sent before acknowledgement (default 64)\n");
    fprintf(stdout, "                                 [cannot be used with -v]\n");
    fprintf(stdout, "  -V, --vary-window              Vary the window size (default no)\n");
    fprintf(stdout, "                                 [cannot be used with -W]\n");
    fprintf(stdout, "  -b, --buffer-num               Use different buffers to perform data transfer (default single)\n");
    fprintf(stdout, "                                 Options: single, multiple\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }
    if (accel_enabled) {
        fprintf(stdout, "  -d, --accelerator  TYPE     use accelerator device buffers, which can be of TYPE `cuda', \n");
        fprintf(stdout, "                              `managed', `openacc', or `rocm' (uses standard host buffers if not specified)\n");
    }
    fprintf(stdout, "  -G, --graph tty,png,pdf        graph output of per"
            " iteration values.\n");
#ifdef _ENABLE_PAPI_
    fprintf(stdout, "  -P, --papi [EVENTS]:[PATH]     Enable PAPI support\n");
    fprintf(stdout, "                                 [EVENTS]       //Comma seperated list of PAPI events\n");
    fprintf(stdout, "                                 [PATH]         //PAPI output file path\n");
#endif
    fprintf(stdout, "  -c, --validation               Enable or disable validation. Disabled by default. \n");
    fprintf(stdout, "  -h, --help                     Print this help\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Note: This benchmark relies on block ordering of the ranks.  Please see\n");
    fprintf(stdout, "        the README for more information.\n");
    fflush(stdout);
}

void print_bad_usage_message (int rank)
{
    if (rank) {
        return;
    }

    if (bad_usage.optarg) {
        fprintf(stderr, "%s [-%c %s]\n\n", bad_usage.message,
                (char)bad_usage.opt, bad_usage.optarg);
    } else {
        fprintf(stderr, "%s [-%c]\n\n", bad_usage.message,
                (char)bad_usage.opt);
    }
    fflush(stderr);

    if (options.bench != ONE_SIDED) {
        print_help_message(rank);
    }
}

void print_help_message (int rank)
{
    if (rank) {
        return;
    }

    if (accel_enabled && (options.bench == PT2PT)) {
        fprintf(stdout, "Usage: %s [options] [SRC DST]\n\n", benchmark_name);
        fprintf(stdout, "SRC and DST are buffer types for the source and destination\n");
        fprintf(stdout, "SRC and DST may be `D', `H', 'MD'or 'MH' which specifies whether\n"
                        "the buffer is allocated on the accelerator device memory, host\n"
                        "memory or using CUDA Unified Memory allocated on device or host respectively for each mpi rank\n\n");
    } else {
        fprintf(stdout, "Usage: %s [options]\n", benchmark_name);
        fprintf(stdout, "Options:\n");
    }

    if (((options.bench == PT2PT) || (options.bench == MBW_MR)) &&
        (LAT_MT != options.subtype) && (LAT_MP != options.subtype)) {
        fprintf(stdout, "  -b, --buffer-num            Use different buffers to perform data transfer (default single)\n");
        fprintf(stdout, "                              Options: single, multiple\n");
    }

    if (accel_enabled && (options.subtype != LAT_MT) && (options.subtype != LAT_MP)) {
        fprintf(stdout, "  -d, --accelerator  TYPE     use accelerator device buffers, which can be of TYPE `cuda', \n");
        fprintf(stdout, "                              `managed', `openacc', or `rocm' (uses standard host buffers if not specified)\n");
    }

    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }

    fprintf(stdout, "  -i, --iterations ITER       set iterations per message size to ITER (default 1000 for small\n");
    fprintf(stdout, "                              messages, 100 for large messages)\n");
    fprintf(stdout, "  -x, --warmup ITER           set number of warmup iterations to skip before timing (default 200)\n");
    fprintf(stdout, "  -I, --imbalance [UG]:EXP:VR introduce imbalance by distribution function (Uniform or Gaussian), \n");
    fprintf(stdout, "                              with expected and variance values in nanoseconds. Examples:\n");
    fprintf(stdout, "                              -I U:1\n");
    fprintf(stdout, "                              -I G:300:100\n");

    if (options.subtype == BW) {
        fprintf(stdout, "  -W, --window-size SIZE      set number of messages to send before synchronization (default 64)\n");
    }

    if ((options.bench == PT2PT)) {
        fprintf(stdout, "  -c, --validation            Enable or disable"
                " validation. Disabled by default. \n");
        fprintf(stdout, "  -u, --validation-warmup ITR Set number of warmup"
                " iterations to skip before timing when validation is enabled"
                " (default 5)\n");
    }

    if (options.bench == COLLECTIVE) {
        fprintf(stdout, "  -f, --full                  print full format listing (MIN/MAX latency and ITERATIONS\n");
        fprintf(stdout, "                              displayed in addition to AVERAGE latency)\n");
        if (options.subtype != NBC) {
            fprintf(stdout, "  -c, --validation            Enable or disable"
                    " validation. Disabled by default. \n");
            fprintf(stdout, "  -u, --validation-warmup ITR Set number of warmup"
                    " iterations to skip before timing when validation is enabled"
                    " (default 5)\n");
        }

        if (options.subtype == NBC ||
                options.subtype == NBC_ALLTOALL ||
                options.subtype == NBC_BCAST ||
                options.subtype == NBC_GATHER ||
                options.subtype == NBC_REDUCE ||
                options.subtype == NBC_SCATTER) {
            fprintf(stdout, "  -t, --num_test_calls CALLS  set the number of MPI_Test() calls during the dummy computation, \n");
            fprintf(stdout, "                              set CALLS to 100, 1000, or any number > 0.\n");
        }

        if (CUDA_KERNEL_ENABLED) {
            fprintf(stdout, "  -r, --cuda-target TARGET    set the compute target for dummy computation\n");
            fprintf(stdout, "                              set TARGET to cpu (default) to execute \n");
            fprintf(stdout, "                              on CPU only, set to gpu for executing kernel \n");
            fprintf(stdout, "                              on the GPU only, and set to both for compute on both.\n");

            fprintf(stdout, "  -a, --array-size SIZE       set the size of arrays to be allocated on device (GPU) \n");
            fprintf(stdout, "                              for dummy compute on device (GPU) (default 32) \n");
        }
    }
    if (LAT_MT == options.subtype) {
        fprintf(stdout, "  -t, --num_threads           SEND:[RECV]  set the sender and receiver number of threads \n");
        fprintf(stdout, "                              min: %d default: (receiver threads: %d sender threads: 1), max: %d.\n", MIN_NUM_THREADS, DEF_NUM_THREADS, MAX_NUM_THREADS);
        fprintf(stdout, "                              Examples: \n");
        fprintf(stdout, "                              -t 4        // receiver threads = 4 and sender threads = 1\n");
        fprintf(stdout, "                              -t 4:6      // sender threads = 4 and receiver threads = 6\n");
        fprintf(stdout, "                              -t 2:       // not defined\n");
    }

    if (LAT_MP == options.subtype) {
        fprintf(stdout, "  -t, --num_processes         SEND:[RECV]  set the sender and receiver number of processes \n");
        fprintf(stdout, "                              min: %d default: (receiver processes: %d sender processes: 1), max: %d.\n",\
                                                       MIN_NUM_PROCESSES, DEF_NUM_PROCESSES, MAX_NUM_PROCESSES);
        fprintf(stdout, "                              Examples: \n");
        fprintf(stdout, "                              -t 4        // receiver processes = 4 and sender processes = 1\n");
        fprintf(stdout, "                              -t 4:6      // sender processes = 4 and receiver processes = 6\n");
        fprintf(stdout, "                              -t 2:       // not defined\n");
    }
    if (options.subtype == GATHER || options.subtype == SCATTER ||
            options.subtype == ALLTOALL || options.subtype == NBC_GATHER ||
            options.subtype == NBC_SCATTER || options.subtype == NBC_ALLTOALL ||
            options.subtype == NBC_BCAST || options.subtype == BCAST ||
            options.bench == PT2PT) {
        fprintf(stdout, "  -D, --ddt [TYPE]:[ARGS]     Enable DDT support\n");
        fprintf(stdout, "                              -D cont                          //Contiguous\n");
        fprintf(stdout, "                              -D vect:[stride]:[block_length]  //Vector\n");
        fprintf(stdout, "                              -D indx:[ddt file path]          //Index\n");
    }
    fprintf(stdout, "  -G, --graph tty,png,pdf    graph output of per"
                        " iteration values.\n");
#ifdef _ENABLE_PAPI_
    fprintf(stdout, "  -P, --papi [EVENTS]:[PATH]     Enable PAPI support\n");
    fprintf(stdout, "                                 [EVENTS]       //Comma seperated list of PAPI events\n");
    fprintf(stdout, "                                 [PATH]         //PAPI output file path\n");
#endif
    fprintf(stdout, "  -h, --help                  print this help\n");
    fprintf(stdout, "  -v, --version               print version info\n");
    fprintf(stdout, "\n");
    fflush(stdout);
}

void print_help_message_get_acc_lat (int rank)
{
    if (rank) {
        return;
    }

    if (bad_usage.optarg) {
        fprintf(stderr, "%s [-%c %s]\n\n", bad_usage.message,
                (char)bad_usage.opt, bad_usage.optarg);
    } else {
        fprintf(stderr, "%s [-%c]\n\n", bad_usage.message,
                (char)bad_usage.opt);
    }
    fflush(stderr);

    fprintf(stdout, "Usage: ./osu_get_acc_latency -w <win_option>  -s < sync_option> [-x ITER] [-i ITER]\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }

    fprintf(stdout, "  -x ITER       number of warmup iterations to skip before timing"
            "(default 100)\n");
    fprintf(stdout, "  -i ITER       number of iterations for timing (default 10000)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "win_option:\n");
    fprintf(stdout, "  create            use MPI_Win_create to create an MPI Window object\n");
    fprintf(stdout, "  allocate          use MPI_Win_allocate to create an MPI Window object\n");
    fprintf(stdout, "  dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "sync_option:\n");
    fprintf(stdout, "  lock              use MPI_Win_lock/unlock synchronizations calls\n");
    fprintf(stdout, "  flush             use MPI_Win_flush synchronization call\n");
    fprintf(stdout, "  flush_local       use MPI_Win_flush_local synchronization call\n");
    fprintf(stdout, "  lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
    fprintf(stdout, "  pscw              use Post/Start/Complete/Wait synchronization calls \n");
    fprintf(stdout, "  fence             use MPI_Win_fence synchronization call\n");
    fprintf(stdout, "\n");

    fflush(stdout);
}

void print_header_one_sided (int rank, enum WINDOW win, enum SYNC sync)
{
    if (rank == 0) {
        switch (options.accel) {
            case CUDA:
                printf(benchmark_header, "-CUDA");
                break;
            case OPENACC:
                printf(benchmark_header, "-OPENACC");
                break;
            case ROCM:
                printf(benchmark_header, "-ROCM");
                break;
            default:
                printf(benchmark_header, "");
                break;
        }
        fprintf(stdout, "# Window creation: %s\n",
                win_info[win]);
        fprintf(stdout, "# Synchronization: %s\n",
                sync_info[sync]);

        switch (options.accel) {
            case CUDA:
            case OPENACC:
            case ROCM:
                fprintf(stdout, "# Rank 0 Memory on %s and Rank 1 Memory on %s\n",
                       'M' == options.src ? "MANAGED (M)" : ('D' == options.src ? "DEVICE (D)" : "HOST (H)"),
                       'M' == options.dst ? "MANAGED (M)" : ('D' == options.dst ? "DEVICE (D)" : "HOST (H)"));
            default:
                if (options.subtype == BW) {
                    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                } else {
                    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                }
                fflush(stdout);
        }
    }
}

void print_version_message (int rank)
{
    if (rank) {
        return;
    }

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case MANAGED:
            printf(benchmark_header, "-CUDA MANAGED");
            break;
        case ROCM:
            printf(benchmark_header, "-ROCM");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fflush(stdout);
}

void print_preamble_nbc (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "\n");

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case MANAGED:
            printf(benchmark_header, "-MANAGED");
            break;
        case ROCM:
            printf(benchmark_header, "-ROCM");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fprintf(stdout, "# Overall = Coll. Init + Compute + MPI_Test + MPI_Wait\n\n");

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overlap(%)");

    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overlap(%)");
    }

    if (options.validate) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Validation");
    }
    if (options.omb_enable_ddt) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Transmit Size");
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}

void display_nbc_params()
{
    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Comm.(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");

    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");
    }
}

void print_preamble (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "\n");

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case ROCM:
            printf(benchmark_header, "-ROCM");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
    } else {
        fprintf(stdout, "# Avg Latency(us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
        fprintf(stdout, "%*s", 12, "Iterations");
    }

    if (options.validate)
        fprintf(stdout, "%*s", FIELD_WIDTH, "Validation");
    if (options.omb_enable_ddt) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Transmit Size");
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}

void check_mem_limit(int numprocs)
{
    int rank = 0;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (options.subtype == GATHER ||
            options.subtype == ALLTOALL ||
            options.subtype == SCATTER ||
            options.subtype == NBC_GATHER ||
            options.subtype == NBC_ALLTOALL ||
            options.subtype == NBC_SCATTER) {
        if ((options.max_message_size * numprocs) > options.max_mem_limit) {
            options.max_message_size = options.max_mem_limit / numprocs;
            if (0 == rank) {
                fprintf(stderr, "Warning! Limiting max message size to: %zu. "
                        "Increase -M, --mem-limit for higher message sizes.",
                        options.max_message_size);
            }
        }
    } else if (options.subtype == REDUCE ||
            options.subtype == BCAST ||
            options.subtype == NBC_REDUCE ||
            options.subtype == NBC_BCAST  ||
            options.subtype == REDUCE_SCATTER ||
            options.subtype == NBC_REDUCE_SCATTER) {
        if (options.max_message_size > options.max_mem_limit) {
            if (0 == rank) {
                fprintf(stderr, "Warning! Limiting max message size to: %zu"
                        "Increase -M, --mem-limit for higher message sizes.",
                        options.max_message_size);
            }
            options.max_message_size = options.max_mem_limit;
        }
    }
}

double calculate_and_print_stats(int rank, int size, int numprocs, double timer,
                               double latency, double test_time,
                               double cpu_time, double wait_time,
                               double init_time, int errors)
{
    double test_total   = (test_time * 1e6) / options.iterations;
    double tcomp_total  = (cpu_time * 1e6) / options.iterations;
    double overall_time = (timer * 1e6) / options.iterations;
    double wait_total   = (wait_time * 1e6) / options.iterations;
    double init_total   = (init_time * 1e6) / options.iterations;
    double avg_comm_time   = latency;
    double min_comm_time = latency, max_comm_time = latency;

    if (rank != 0) {
        MPI_CHECK(MPI_Reduce(&test_total, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&avg_comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&overall_time, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&tcomp_total, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&wait_total, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&init_total, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &min_comm_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD));
    } else {
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &min_comm_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                   MPI_COMM_WORLD));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    /* Overall Time (Overlapped) */
    overall_time = overall_time/numprocs;
    /* Computation Time */
    tcomp_total = tcomp_total/numprocs;
    /* Time taken by MPI_Test calls */
    test_total = test_total/numprocs;
    /* Pure Communication Time */
    avg_comm_time = avg_comm_time/numprocs;
    /* Time for MPI_Wait() call */
    wait_total = wait_total/numprocs;
    /* Time for the NBC call */
    init_total = init_total/numprocs;


    print_stats_nbc(rank, size, overall_time, tcomp_total, avg_comm_time,
                    min_comm_time, max_comm_time, wait_total, init_total,
                    test_total, errors);
    return overall_time;

}

void print_stats_nbc (int rank, int size, double overall_time, double cpu_time,
                      double avg_comm_time, double min_comm_time,
                      double max_comm_time, double wait_time, double init_time,
                      double test_time, int errors)
{
    if (rank) {
        return;
    }

    double overlap;

    /* Note : cpu_time received in this function includes time for
       *      dummy compute as well as test calls so we will subtract
       *      the test_time for overlap calculation as test is an
       *      overhead
       */

    overlap = MAX(0, 100 - (((overall_time - (cpu_time - test_time)) / avg_comm_time) * 100));

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    } else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f",
                FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time),
                FIELD_WIDTH, FLOAT_PRECISION, init_time,
                FIELD_WIDTH, FLOAT_PRECISION, test_time,
                FIELD_WIDTH, FLOAT_PRECISION, wait_time,
                FIELD_WIDTH, FLOAT_PRECISION, avg_comm_time,
                FIELD_WIDTH, FLOAT_PRECISION, min_comm_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_comm_time,
                FIELD_WIDTH, FLOAT_PRECISION, overlap);
    } else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time));
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_comm_time);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overlap);
    }

    if (options.validate) {
        fprintf(stdout, "%*s", FIELD_WIDTH, VALIDATION_STATUS(errors));
    }
    if (!options.omb_enable_ddt) {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void print_stats (int rank, int size, double avg_time, double min_time, double max_time)
{
    if (rank) {
        return;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
    } else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    }
    if (!options.omb_enable_ddt) {
        fprintf(stdout, "\n");
    }
    fflush(stdout);
}

void print_stats_validate(int rank, int size, double avg_time, double min_time,
            double max_time, int errors)
{
    if (rank) {
        return;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
    } else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    }
    fprintf(stdout, "%*s", FIELD_WIDTH, VALIDATION_STATUS(errors));
    if (!options.omb_enable_ddt) {
        fprintf(stdout, "\n");
    }
    fflush(stdout);
}

void omb_ddt_append_stats(size_t omb_ddt_transmit_size)
{
    int rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (rank) {
        return;
    }
    if (options.omb_enable_ddt) {
        fprintf(stdout, "%*zu\n", FIELD_WIDTH, omb_ddt_transmit_size);
    }
}

void set_buffer_pt2pt (void * buffer, int rank, enum accel_type type, int data,
                       size_t size)
{
    char buf_type = 'H';

    if (options.bench == MBW_MR) {
        buf_type = (rank < options.pairs) ? options.src : options.dst;
    } else {
        buf_type = (rank == 0) ? options.src : options.dst;
    }

    switch (buf_type) {
        case 'H':
            memset(buffer, data, size);
            break;
        case 'D':
        case 'M':
#ifdef _ENABLE_OPENACC_
            if (type == OPENACC) {
                size_t i;
                char * p = (char *)buffer;
                #pragma acc parallel loop deviceptr(p)
                for (i = 0; i < size; i++) {
                    p[i] = data;
                }
                break;
            } else
#endif
#ifdef _ENABLE_CUDA_
            {
                CUDA_CHECK(cudaMemset(buffer, data, size));
            }
#endif
#ifdef _ENABLE_ROCM_
            {
                ROCM_CHECK(hipMemset(buffer, data, size));
            }
#endif
            break;
    }
}

void set_buffer (void * buffer, enum accel_type type, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)buffer;
#endif
    switch (type) {
        case NONE:
            memset(buffer, data, size);
            break;
        case CUDA:
        case MANAGED:
#ifdef _ENABLE_CUDA_
            CUDA_CHECK(cudaMemset(buffer, data, size));
#endif
            break;
        case OPENACC:
#ifdef _ENABLE_OPENACC_
            #pragma acc parallel loop deviceptr(p)
            for (i = 0; i < size; i++) {
                p[i] = data;
            }
#endif
            break;
        case ROCM:
#ifdef _ENABLE_ROCM_
            ROCM_CHECK(hipMemset(buffer, data, size));
#endif
            break;
        default:
            break;
    }
}

void set_buffer_validation(void* s_buf, void* r_buf, size_t size,
                           enum accel_type type, int iter)
{
    void *temp_r_buffer = NULL;
    void *temp_s_buffer = NULL;
    int rank = 0;
    char buf_type = 'H';
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    switch (options.bench)
    {
        case PT2PT:
        case MBW_MR:
            {
                int num_elements = size / sizeof(char);
                temp_r_buffer = malloc(size);
                temp_s_buffer = malloc(size);
                char* temp_char_s_buffer = (char*) temp_s_buffer;
                char* temp_char_r_buffer = (char*) temp_r_buffer;
                register int i;
                for (i = 0; i < num_elements; i++) {
                    temp_char_s_buffer[i] = (CHAR_VALIDATION_MULTIPLIER * (i +
                                1) + size + iter) % CHAR_RANGE;
                }
                for (i = 0; i < num_elements; i++) {
                    temp_char_r_buffer[i] = 0;
                }
                if (options.bench == MBW_MR) {
                    buf_type = (rank < options.pairs) ? options.src : options.dst;
                } else {
                    buf_type = (rank == 0) ? options.src : options.dst;
                }
                switch (buf_type) {
                    case 'H':
                        memcpy((void *)s_buf, (void *)temp_s_buffer, size);
                        memcpy((void *)r_buf, (void *)temp_r_buffer, size);
                        break;
                    case 'D':
                    case 'M':
#ifdef _ENABLE_OPENACC_
                        if (type == OPENACC) {
                            size_t i;
                            char * p = (char *)s_buf;
			    #pragma acc parallel loop deviceptr(p)
                            for (i = 0; i < num_elements; i++) {
                                p[i] = temp_char_s_buffer[i];
                            }
                            p = (char *)r_buf;
			    #pragma acc parallel loop deviceptr(p)
                            for (i = 0; i < num_elements; i++) {
                                p[i] = temp_char_r_buffer[i];
                            }
                            break;
                        } else
#endif
#ifdef _ENABLE_CUDA_
                        {
                            CUDA_CHECK(cudaMemcpy((void *)s_buf,
                                        (void *)temp_s_buffer, size,
                                        cudaMemcpyHostToDevice));
                            CUDA_CHECK(cudaMemcpy((void *)r_buf,
                                        (void *)temp_r_buffer, size,
                                        cudaMemcpyHostToDevice));
                            CUDA_CHECK(cudaDeviceSynchronize());
                        }
#endif
#ifdef _ENABLE_ROCM_
                        {
                            ROCM_CHECK(hipMemcpy((void *)s_buf,
                                        (void *)temp_s_buffer, size,
                                        hipMemcpyHostToDevice));
                            ROCM_CHECK(hipMemcpy((void *)r_buf,
                                        (void *)temp_r_buffer, size,
                                        hipMemcpyHostToDevice));
                            ROCM_CHECK(hipDeviceSynchronize());
                        }
#endif
                        break;
                }
                free(temp_s_buffer);
                free(temp_r_buffer);
            }
            break;
        case COLLECTIVE:
            {
                switch(options.subtype) {
                    case ALLTOALL:
                    case NBC_ALLTOALL:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            set_buffer_char(s_buf, 1, size, rank, numprocs,
                                    type, iter);
                            set_buffer_char(r_buf, 0, size, rank, numprocs,
                                    type, iter);
                        }
                        break;
                    case GATHER:
                    case NBC_GATHER:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            set_buffer_char(s_buf, 1, size, rank * numprocs, 1,
                                    type, iter);
                            if (0 == rank) {
                                set_buffer_char(r_buf, 0, size, rank, numprocs,
                                        type, iter);
                            }
                        }
                        break;
                    case REDUCE:
                    case NBC_REDUCE:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            set_buffer_int(s_buf, 1, size, iter, options.accel);
                            set_buffer_int(r_buf, 0, size, iter, options.accel);
                        break;
                    }
                    case SCATTER:
                    case NBC_SCATTER:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            if (0 == rank) {
                                set_buffer_char(s_buf, 1, size, rank, numprocs,
                                        type, iter);
                            }
                            set_buffer_char(r_buf, 0, size, rank * numprocs, 1,
                                    type, iter);
                        }
                        break;
                    case REDUCE_SCATTER:
                    case NBC_REDUCE_SCATTER:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            set_buffer_float(s_buf, 1, size, iter,
                                    options.accel);
                            set_buffer_float(r_buf, 0, size / numprocs + 1,
                                    iter, options.accel);
                        }
                        break;
                    case BCAST:
                    case NBC_BCAST:
                        {
                            int rank, numprocs;
                            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                            if (0 == rank) {
                                set_buffer_char(s_buf, 1, size, 1, 1, type,
                                        iter);
                            } else {
                                set_buffer_char(s_buf, 0, size, 1, 1, type,
                                        iter);
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
            break;
        default:
            break;
    }
}

void set_buffer_float (float* buffer, int is_send_buf, size_t size, int iter,
                       enum accel_type type)
{
    if (NULL == buffer) {
        return;
    }

    int i = 0, j = 0;
    int num_elements = size;
    float *temp_buffer = malloc(size * sizeof(float));
    if (is_send_buf) {
        for (i = 0; i < num_elements; i++) {
            j = (i % 100);
            temp_buffer[i] = (j + 1) * (iter + 1) * 1.0;
        }
    } else {
        for (i = 0; i < num_elements; i++) {
            temp_buffer[i] = 0.0;
        }
    }
    switch (type) {
        case NONE:
            memcpy((void *)buffer, (void *)temp_buffer, size * sizeof(float));
            break;
        case CUDA:
        case MANAGED:
#ifdef _ENABLE_CUDA_
            CUDA_CHECK(cudaMemcpy((void *)buffer, (void *)temp_buffer,
                       size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
            break;
        default:
            break;
    }
    free(temp_buffer);
}

void set_buffer_int (int* buffer, int is_send_buf, size_t size, int iter,
                       enum accel_type type)
{
    if (NULL == buffer) {
        return;
    }

    int i = 0, j = 0;
    int num_elements = size;
    int *temp_buffer = malloc(size * sizeof(float));
    if (is_send_buf) {
        for (i = 0; i < num_elements; i++) {
            j = (i % 100);
            temp_buffer[i] = (j + 1) * (iter + 1);
        }
    } else {
        for (i = 0; i < num_elements; i++) {
            temp_buffer[i] = 0;
        }
    }
    switch (type) {
        case NONE:
            memcpy((void *)buffer, (void *)temp_buffer, size * sizeof(float));
            break;
        case CUDA:
        case MANAGED:
#ifdef _ENABLE_CUDA_
            CUDA_CHECK(cudaMemcpy((void *)buffer, (void *)temp_buffer,
                       size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
            break;
        default:
            break;
    }
    free(temp_buffer);
}

void set_buffer_char (char * buffer, int is_send_buf, size_t size, int rank,
                      int num_procs, enum accel_type type, int iter)
{
    if (NULL == buffer) {
        return;
    }

    int num_elements = size / sizeof(char);
    int i, j;
    char *temp_buffer = malloc(size * num_procs);
    if (is_send_buf) {
        for (i = 0; i < num_procs; i++) {
            for (j = 0; j < num_elements; j++) {
                temp_buffer[i * num_elements + j] = (rank * num_procs + i +
                        ((iter + 1) * (rank * num_procs + 1) * (i + 1))) %
                        (1<<8);
            }
        }
    } else {
        for (i = 0; i < num_procs * num_elements; i++) {
            temp_buffer[i] = 0;
        }
    }
    switch (type) {
        case NONE:
            memcpy((void *)buffer, (void *)temp_buffer, size * num_procs);
            break;
        case CUDA:
        case MANAGED:
#ifdef _ENABLE_CUDA_
            CUDA_CHECK(cudaMemcpy((void *)buffer, (void *)temp_buffer,
                       size * num_procs, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
            break;
        default:
            break;
    }
    free(temp_buffer);
}

uint8_t validate_data(void* r_buf, size_t size, int num_procs,
                      enum accel_type type, int iter)
{
    void *temp_r_buf = NULL;
    int rank = 0;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    switch (options.bench)
    {
        case PT2PT:
        case MBW_MR:
            {
                register int i = 0;
                int num_elements = size / sizeof(char);
                temp_r_buf = malloc(size);
                char* temp_char_r_buf = (char*) temp_r_buf;
                char* expected_buffer = malloc(size);
                char buf_type = 'H';

                if (options.bench == MBW_MR) {
                    buf_type = (rank < options.pairs) ? options.src : options.dst;
                } else {
                    buf_type = (rank == 0) ? options.src : options.dst;
                }
                switch (buf_type) {
                    case 'H':
                        memcpy((void *)temp_char_r_buf, (void *)r_buf, size);
                        break;
                    case 'D':
                    case 'M':
#ifdef _ENABLE_OPENACC_
                        if (type == OPENACC) {
                            size_t i;
			    char * p = (char *)r_buf;
			    #pragma acc parallel loop deviceptr(p)
			    for (i = 0; i < num_elements; i++) {
				temp_char_r_buf[i] = p[i];
                            }
                            break;
                        } else
#endif
#ifdef _ENABLE_CUDA_
                        {
                        CUDA_CHECK(cudaMemcpy((void *)temp_char_r_buf, (void *)r_buf,
                                    size, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaDeviceSynchronize());
                        }
#endif
#ifdef _ENABLE_ROCM_
                        {
                        ROCM_CHECK(hipMemcpy((void *)temp_char_r_buf, (void *)r_buf,
                                    size, hipMemcpyDeviceToHost));
                        ROCM_CHECK(hipDeviceSynchronize());
                        }
#endif
                        break;
                }
                for (i = 0; i < num_elements; i++) {
                    expected_buffer[i] = (CHAR_VALIDATION_MULTIPLIER * (i + 1) +
                            size + iter) % CHAR_RANGE;
                }
                if (memcmp(temp_char_r_buf, expected_buffer, num_elements)) {
                    free(temp_r_buf);
                    return 1;
                }
                free(temp_r_buf);
                return 0;
            }
            break;
        case COLLECTIVE:
            switch (options.subtype)
            {
            case REDUCE:
            case NBC_REDUCE:
                {
                    int numprocs;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                    return validate_reduction(r_buf, size, iter, numprocs, options.accel);
                }
                break;
            case ALLTOALL:
            case NBC_ALLTOALL:
                {
                    int numprocs, rank;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                    return validate_collective(r_buf, size, rank, numprocs,
                            type, iter);
                }
                break;
            case GATHER:
            case NBC_GATHER:
                {
                    int numprocs;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                    return validate_collective(r_buf, size, 0, numprocs, type,
                            iter);
                }
                break;

            case SCATTER:
            case NBC_SCATTER:
                {
                    int numprocs, rank;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                    return validate_collective(r_buf, size, rank, 1, type,
                            iter);
                }
                break;
            case REDUCE_SCATTER:
            case NBC_REDUCE_SCATTER:
                {
                    int numprocs;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
                    return validate_reduction(r_buf, size, iter, numprocs,
                            options.accel);
                }
                break;
            case BCAST:
            case NBC_BCAST:
                {
                    return validate_collective(r_buf, size, 1, 1, type, iter);
                }
                break;

            default:
                break;
            }
            break;
        default:
            break;

    }
    return 1;
}

int validate_reduce_scatter(float *buffer, size_t size, int* recvcounts,
                            int rank, int num_procs, enum accel_type type,
                            int iter)
{
    int i = 0, j = 0, k = 0, errors = 0;
    float *expected_buffer = malloc(size * sizeof(float));
    float *temp_buffer = malloc(size * sizeof(float));

    switch (type) {
        case NONE:
            memcpy((void *)temp_buffer, (void *)buffer, size * sizeof(float));
            break;
#ifdef _ENABLE_CUDA_
        case CUDA:
        case MANAGED:
            CUDA_CHECK(cudaMemcpy((void *)temp_buffer, (void *)buffer, size *
                        sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
#endif
        default:
            break;
    }

    i = 0;
    for (k = 0; k < rank; k++) {
        i += recvcounts[k] + 1;
    }
    for (i = i; i < recvcounts[k]; i++) {
        j = (i % 100);
        expected_buffer[i] = (j + 1) * (iter + 1) * 1.0 * num_procs;
        if (abs(temp_buffer[i] - expected_buffer[i]) > ERROR_DELTA) {
            errors = 1;
            break;
        }
    }
    free(expected_buffer);
    free(temp_buffer);
    return errors;

}

int validate_reduction(int *buffer, size_t size, int iter, int num_procs,
                       enum accel_type type)
{
    int i = 0, j = 0, errors = 0;
    int *expected_buffer = malloc(size * sizeof(int));
    int *temp_buffer = malloc(size * sizeof(int));
    int num_elements = size;

    switch (type) {
        case NONE:
            memcpy((void *)temp_buffer, (void *)buffer, size * sizeof(int));
            break;
#ifdef _ENABLE_CUDA_
        case CUDA:
        case MANAGED:
            CUDA_CHECK(cudaMemcpy((void *)temp_buffer, (void *)buffer, size *
                        sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
#endif
        default:
            break;
    }

    for (i = 0; i < num_elements; i++) {
        j = (i % 100);
        expected_buffer[i] = (j + 1) * (iter + 1) * num_procs;

        if (abs(temp_buffer[i] - expected_buffer[i]) > ERROR_DELTA) {
printf("ERROR[%i]: temp_buffer[]=%i expected_buffer=%i\n", i, temp_buffer[i], expected_buffer[i]);
            errors = 1;
            break;
        }
    }
    free(expected_buffer);
    free(temp_buffer);
    return errors;
}

int validate_collective(char *buffer, size_t size, int value1, int value2,
                        enum accel_type type, int itr)
{
    int i = 0, j = 0, errors = 0;
    char *expected_buffer = malloc(size * value2);
    char *temp_buffer = malloc(size* value2);
    int num_elements = size / sizeof(char);

    switch (type) {
        case NONE:
            memcpy((void *)temp_buffer, (void *)buffer, size * value2);
            break;
#ifdef _ENABLE_CUDA_
        case CUDA:
        case MANAGED:
            CUDA_CHECK(cudaMemcpy((void *)temp_buffer, (void *)buffer, size *
                        value2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
#endif
        default:
            break;
    }

    for (i = 0; i < value2; i++) {
        for (j = 0; j < num_elements; j++) {
            expected_buffer[i * num_elements + j] = (i * value2 + value1 +
                            ((itr + 1) * (value1 + 1) * (i * value2 + 1))) %
                            (1<<8);
            }
        }
        if (memcmp(temp_buffer, expected_buffer, size * value2) != 0) {
            errors = 1;
        }
    free(expected_buffer);
    free(temp_buffer);
    return errors;
}

int allocate_memory_coll (void ** buffer, size_t size, enum accel_type type)
{
    if (options.target == CPU || options.target == BOTH) {
        allocate_host_arrays();
    }

    size_t alignment = sysconf(_SC_PAGESIZE);

    switch (type) {
        case NONE:
            return posix_memalign(buffer, alignment, size);
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaMalloc(buffer, size));
            return 0;
        case MANAGED:
            CUDA_CHECK(cudaMallocManaged(buffer, size, cudaMemAttachGlobal));
            return 0;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            *buffer = acc_malloc(size);
            if (NULL == *buffer) {
                return 1;
            } else {
                return 0;
            }
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
            ROCM_CHECK(hipMalloc(buffer, size));
            return 0;
#endif
        default:
            return 1;
    }
}

int allocate_device_buffer (char ** buffer)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
             CUDA_CHECK(cudaMalloc((void **)buffer, options.max_message_size));
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            *buffer = acc_malloc(options.max_message_size);
            if (NULL == *buffer) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
             ROCM_CHECK(hipMalloc((void **)buffer, options.max_message_size));
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;
    }

    return 0;
}

int allocate_device_buffer_one_sided (char ** buffer, size_t size)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaMalloc((void **)buffer, size));
            break;
        case MANAGED:
            CUDA_CHECK(cudaMallocManaged((void **)buffer, size, cudaMemAttachGlobal));
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            *buffer = acc_malloc(size);
            if (NULL == *buffer) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
             ROCM_CHECK(hipMalloc((void **)buffer, size));
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;
    }

    return 0;
}

int allocate_managed_buffer (char ** buffer)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaMallocManaged((void **)buffer, options.max_message_size, cudaMemAttachGlobal));
	    break;
#endif
        default:
            fprintf(stderr, "Could not allocate managed/unified memory\n");
            return 1;

    }
    return 0;
}

int allocate_managed_buffer_size (char ** buffer, size_t size)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            CUDA_CHECK(cudaMallocManaged((void **)buffer, size, cudaMemAttachGlobal));
	    break;
#endif
        default:
            fprintf(stderr, "Could not allocate managed memory\n");
            return 1;

    }
    return 0;
}

int allocate_memory_pt2pt_mul (char ** sbuf, char ** rbuf, int rank, int pairs)
{
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    if (rank < pairs) {
        if ('D' == options.src) {
            if (allocate_device_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }

            if (allocate_device_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }
        } else if ('M' == options.src) {
            if (allocate_managed_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }

            if (allocate_managed_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }
        } else {
            if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            memset(*sbuf, 0, options.max_message_size);
            memset(*rbuf, 0, options.max_message_size);
        }
    } else {
        if ('D' == options.dst) {
            if (allocate_device_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }

            if (allocate_device_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }
        } else if ('M' == options.dst) {
            if (allocate_managed_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }

            if (allocate_managed_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }
        } else {
            if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }
            memset(*sbuf, 0, options.max_message_size);
            memset(*rbuf, 0, options.max_message_size);
        }
    }

    return 0;
}

int allocate_memory_pt2pt_mul_size (char ** sbuf, char ** rbuf, int rank, int pairs, size_t allocate_size)
{
    size_t size;
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    if (allocate_size == 0) {
        size = 1;
    } else {
        size = allocate_size;
    }

    if (rank < pairs) {
        if ('D' == options.src) {
            if (allocate_device_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }

            if (allocate_device_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }
        } else if ('M' == options.src) {
            if (allocate_managed_buffer_size(sbuf, size)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }

            if (allocate_managed_buffer_size(rbuf, size)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }
        } else {
            if (posix_memalign((void**)sbuf, align_size, size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            if (posix_memalign((void**)rbuf, align_size, size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            memset(*sbuf, 0, size);
            memset(*rbuf, 0, size);
        }
    } else {
        if ('D' == options.dst) {
            if (allocate_device_buffer(sbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }

            if (allocate_device_buffer(rbuf)) {
                fprintf(stderr, "Error allocating cuda memory\n");
                return 1;
            }
        } else if ('M' == options.dst) {
            if (allocate_managed_buffer_size(sbuf, size)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }

            if (allocate_managed_buffer_size(rbuf, size)) {
                fprintf(stderr, "Error allocating cuda unified memory\n");
                return 1;
            }
        } else {
            if (posix_memalign((void**)sbuf, align_size, size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }

            if (posix_memalign((void**)rbuf, align_size, size)) {
                fprintf(stderr, "Error allocating host memory\n");
                return 1;
            }
            memset(*sbuf, 0, size);
            memset(*rbuf, 0, size);
        }
    }

    return 0;
}

int allocate_memory_pt2pt (char ** sbuf, char ** rbuf, int rank)
{
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.src) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
        case 1:
            if ('D' == options.dst) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.dst) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
    }

    return 0;
}

int allocate_memory_pt2pt_size (char ** sbuf, char ** rbuf, int rank, size_t allocate_size)
{
    size_t size;
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    if (allocate_size == 0) {
        size = 1;
    } else {
        size = allocate_size;
    }

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.src) {
                if (allocate_managed_buffer_size(sbuf, size)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer_size(rbuf, size)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
        case 1:
            if ('D' == options.dst) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.dst) {
                if (allocate_managed_buffer_size(sbuf, size)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer_size(rbuf, size)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
    }

    return 0;
}

void allocate_memory_one_sided(int rank, char **user_buf,
        char **win_base, size_t size, enum WINDOW type, MPI_Win *win)
{
    int page_size;
    int purehost = 0;
    int mem_on_dev = 0;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if ('H' == options.src && 'H' == options.dst) {
        purehost = 1;
    }

    if (rank == 0) {
        mem_on_dev = ('H' == options.src) ? 0 : 1;
    } else {
        mem_on_dev = ('H' == options.dst) ? 0 : 1;
    }

    /* always allocate device buffers explicitly since most MPI libraries do not
     * support allocating device buffers during window creation */
    if (mem_on_dev) {
        CHECK(allocate_device_buffer_one_sided(user_buf, size));
        set_device_memory(*user_buf, 'a', size);
        CHECK(allocate_device_buffer_one_sided(win_base, size));
        set_device_memory(*win_base, 'a', size);
    } else {
        CHECK(posix_memalign((void **)user_buf, page_size, size));
        memset(*user_buf, 'a', size);
        /* only explicitly allocate buffer for win_base when NOT using MPI_Win_allocate */
        if (type != WIN_ALLOCATE) {
            CHECK(posix_memalign((void **)win_base, page_size, size));
            memset(*win_base, 'a', size);
        }
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if (rank == 0) {
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            } else {
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (purehost) {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, (void*) win_base, win));
            } else {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

size_t omb_ddt_assign(MPI_Datatype *datatype, MPI_Datatype base_datatype,
        size_t size)
{
/* Since all the benchmarks that supports ddt currently use char, count is
 * equal to char. This should be modified in the future if ddt support
 * for other bencharks are included.
 */
    size_t count = size;
    size_t transmit_size = 0;
    FILE *fp = NULL;
    char line[OMB_DDT_FILE_LINE_MAX_LENGTH];
    char *token;
    int i = 0;
    int block_lengths[OMB_DDT_INDEXED_MAX_LENGTH] = {0};
    int displacements[OMB_DDT_INDEXED_MAX_LENGTH] = {0};

    if (0 == options.omb_enable_ddt) {
        return size;
    }
    switch (options.ddt_type) {
        case OMB_DDT_CONTIGUOUS:
            MPI_CHECK(MPI_Type_contiguous(count, base_datatype, datatype));
            MPI_CHECK(MPI_Type_commit(datatype));
            transmit_size = count;
            break;
        case OMB_DDT_VECTOR:
            MPI_CHECK(MPI_Type_vector(count /
                        options.ddt_type_parameters.stride,
                        options.ddt_type_parameters.block_length,
                        options.ddt_type_parameters.stride, base_datatype,
                        datatype));
            MPI_CHECK(MPI_Type_commit(datatype));
            transmit_size = (count / options.ddt_type_parameters.stride) *
                options.ddt_type_parameters.block_length;
            break;
        case OMB_DDT_INDEXED:
            fp = fopen(options.ddt_type_parameters.filepath, "r");
            OMB_CHECK_NULL_AND_EXIT(fp, "Unable to open ddt indexed file.\n");
            transmit_size = 0;
            while (fgets(line, OMB_DDT_FILE_LINE_MAX_LENGTH, fp)) {
                if ('#' == line[0]) {
                    continue;
                }
                token = strtok(line, ",");
                if (NULL != token) {
                    displacements[i] = atoi(token);
                }
                token = strtok(NULL, ",");
                if (NULL != token) {
                    block_lengths[i] = atoi(token);
                    transmit_size += block_lengths[i];
                }
                i++;
            }
            fclose(fp);
            MPI_CHECK(MPI_Type_indexed(i-1,
                    block_lengths, displacements, base_datatype, datatype));
            MPI_CHECK(MPI_Type_commit(datatype));
            break;
    }
    return transmit_size;
}

void omb_ddt_free(MPI_Datatype *datatype)
{
    if (0 == options.omb_enable_ddt) {
        return;
    }
    OMB_CHECK_NULL_AND_EXIT(datatype, "Received NULL datatype");
    MPI_CHECK(MPI_Type_free(datatype));
}

size_t omb_ddt_get_size(size_t size)
{
    if (0 == options.omb_enable_ddt) {
        return size;
    }
    return 1;
}

void free_buffer (void * buffer, enum accel_type type)
{
    switch (type) {
        case NONE:
            free(buffer);
            break;
        case MANAGED:
        case CUDA:
#ifdef _ENABLE_CUDA_
            CUDA_CHECK(cudaFree(buffer));
#endif
            break;
        case OPENACC:
#ifdef _ENABLE_OPENACC_
            acc_free(buffer);
#endif
            break;
        case ROCM:
#ifdef _ENABLE_ROCM_
            ROCM_CHECK(hipFree(buffer));
#endif
            break;
    }

    /* Free dummy compute related resources */
    if (CPU == options.target || BOTH == options.target) {
        free_host_arrays();
    }

    if (GPU == options.target || BOTH == options.target) {
#ifdef _ENABLE_CUDA_KERNEL_
        free_device_arrays();
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
    }
}

#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_) || defined(_ENABLE_ROCM_)
int omb_get_local_rank()
{
    char *str = NULL;
    int local_rank = -1;

    if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("MPI_LOCALRANKID")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("SLURM_PROCID")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else {
        fprintf(stderr, "Warning: OMB could not identify the local rank of the process.\n");
        fprintf(stderr, "         This can lead to multiple processes using the same GPU.\n");
        fprintf(stderr, "         Please use the get_local_rank script in the OMB repo for this.\n");
    }

    return local_rank;
}
#endif /* defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_) || defined(_ENABLE_ROCM_) */

int init_accel (void)
{
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
    CUdevice cuDevice;
#endif
#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_) || defined(_ENABLE_ROCM_)
    int local_rank = -1, dev_count = 0;
    int dev_id = 0;

    local_rank = omb_get_local_rank();
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_KERNEL_
        case MANAGED:
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
#ifdef _ENABLE_CUDA_
        case CUDA:
            if (local_rank >= 0) {
                CUDA_CHECK(cudaGetDeviceCount(&dev_count));
                dev_id = local_rank % dev_count;
            }
            CUDA_CHECK(cudaSetDevice(dev_id));

            curesult = cuInit(0);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuDeviceGet(&cuDevice, dev_id);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

#ifdef _ENABLE_CUDA_KERNEL_
            create_cuda_stream();
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            if (local_rank >= 0) {
                dev_count = acc_get_num_devices(acc_device_not_host);
                assert(dev_count > 0);
                dev_id = local_rank % dev_count;
            }

            acc_set_device_num (dev_id, acc_device_not_host);
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
            if (local_rank >= 0) {
                ROCM_CHECK(hipGetDeviceCount(&dev_count));
                dev_id = local_rank % dev_count;
            }
            ROCM_CHECK(hipSetDevice(dev_id));
            break;
#endif
        default:
            fprintf(stderr, "Invalid device type, should be cuda, openacc, or rocm. "
                    "Check configure time options to verify that support for chosen "
                    "device type is enabled.\n");
            return 1;
    }

    return 0;
}

int cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_KERNEL_
        case MANAGED:
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
#ifdef _ENABLE_CUDA_
        case CUDA:
            /* Reset the device to release all resources */
#ifdef _ENABLE_CUDA_KERNEL_
            destroy_cuda_stream();
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
            CUDA_CHECK(cudaDeviceReset());
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            acc_shutdown(acc_device_nvidia);
            break;
#endif
#ifdef _ENABLE_ROCM_
        case ROCM:
            ROCM_CHECK(hipDeviceReset());
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

#ifdef _ENABLE_CUDA_KERNEL_
void free_device_arrays()
{
    if (is_alloc) {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));

        is_alloc = 0;
    }
}
#endif

void free_host_arrays()
{
    int i = 0;

    if (x) {
        free(x);
    }
    if (y) {
        free(y);
    }

    if (a) {
        for (i = 0; i < DIM; i++) {
            free(a[i]);
        }
        free(a);
    }

    x = NULL;
    y = NULL;
    a = NULL;
}

void free_memory (void * sbuf, void * rbuf, int rank)
{
    switch (rank) {
        case 0:
            if ('D' == options.src || 'M' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            } else {
                if (sbuf) {
                    free(sbuf);
                }
                if (rbuf) {
                    free(rbuf);
                }
            }
            break;
        case 1:
            if ('D' == options.dst || 'M' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            } else {
                if (sbuf) {
                    free(sbuf);
                }
                if (rbuf) {
                    free(rbuf);
                }
            }
            break;
    }
}

void free_memory_pt2pt_mul (void * sbuf, void * rbuf, int rank, int pairs)
{
    if (rank < pairs) {
        if ('D' == options.src || 'M' == options.src) {
            free_device_buffer(sbuf);
            free_device_buffer(rbuf);
        } else {
            free(sbuf);
            free(rbuf);
        }
    } else {
        if ('D' == options.dst || 'M' == options.dst) {
            free_device_buffer(sbuf);
            free_device_buffer(rbuf);
        } else {
            free(sbuf);
            free(rbuf);
        }
    }
}

void free_memory_one_sided (void *user_buf, void *win_baseptr, enum WINDOW win_type, MPI_Win win, int rank)
{
    MPI_CHECK(MPI_Win_free(&win));
    /* if MPI_Win_allocate is specified, win_baseptr would be freed by MPI_Win_free,
     * so only need to free the user_buf */
    if (win_type == WIN_ALLOCATE) {
        free_memory(user_buf, NULL, rank);
    } else {
        free_memory(user_buf, win_baseptr, rank);
    }
}

double dummy_compute(double seconds, MPI_Request* request)
{
    double test_time = 0.0;

    test_time = do_compute_and_probe(seconds, request);

    return test_time;
}

#ifdef _ENABLE_CUDA_KERNEL_
void create_cuda_stream()
{
    CUDA_CHECK(cudaStreamCreate(&um_stream));
}

void destroy_cuda_stream()
{
    CUDA_CHECK(cudaStreamDestroy(um_stream));
}

void create_cuda_event()
{
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}

void destroy_cuda_event()
{
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void event_record_start()
{
    CUDA_CHECK(cudaEventRecord(start, um_stream));
}

void event_record_stop()
{
    CUDA_CHECK(cudaEventRecord(stop, um_stream));
}

void event_elapsed_time(float * t_elapsed)
{

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(t_elapsed, start, stop));
}

void synchronize_device()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

void synchronize_stream()
{
    CUDA_CHECK(cudaStreamSynchronize(um_stream));
}

void prefetch_data(char *buf, size_t length, int devid)
{
    CUDA_CHECK(cudaMemPrefetchAsync(buf, length, devid, um_stream));
}

void touch_managed(char *buf, size_t length)
{
    call_touch_managed_kernel(buf, length, &um_stream);
}

void launch_empty_kernel(char *buf, size_t length)
{
    call_empty_kernel(buf, length, &um_stream);
}

void do_compute_gpu(double seconds)
{
    double time_elapsed = 0.0, t1 = 0.0, t2 = 0.0;

    {
        t1 = MPI_Wtime();

        /* Execute Dummy Kernel on GPU if set by user */
        if (options.target == BOTH || options.target == GPU) {
            {
                CUDA_CHECK(cudaStreamCreate(&stream));
                call_kernel(A, d_x, d_y, options.device_array_size, &stream);
            }
        }

        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
}
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

void compute_on_host()
{
    int i = 0, j = 0;
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            x[i] = x[i] + a[i][j]*a[j][i] + y[j];
}

static inline void do_compute_cpu(double target_seconds)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_seconds) {
        t1 = MPI_Wtime();
        compute_on_host();
        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
    if (DEBUG) {
        fprintf(stderr, "time elapsed = %f\n", (time_elapsed * 1e6));
    }
}

double do_compute_and_probe(double seconds, MPI_Request* request)
{
    double t1 = 0.0, t2 = 0.0;
    double test_time = 0.0;
    int num_tests = 0;
    double target_seconds_for_compute = 0.0;
    int flag = 0;
    MPI_Status status;

    if (options.num_probes) {
        target_seconds_for_compute = (double) seconds/options.num_probes;
        if (DEBUG) {
            fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
        }
    } else {
        target_seconds_for_compute = seconds;
        if (DEBUG) {
            fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU only */
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        } else {
            do_compute_gpu(target_seconds_for_compute);
        }
    } else if (options.target == BOTH) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU and CPU*/
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
                do_compute_cpu(target_seconds_for_compute);
            }
        } else {
            do_compute_gpu(target_seconds_for_compute);
            do_compute_cpu(target_seconds_for_compute);
        }
    } else
#endif
    if (options.target == CPU) {
        if (options.num_probes) {
            num_tests = 0;
            while (num_tests < options.num_probes) {
                do_compute_cpu(target_seconds_for_compute);
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        } else {
            do_compute_cpu(target_seconds_for_compute);
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU || options.target == BOTH) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
#endif

    return test_time;
}

void allocate_host_arrays()
{
    int i=0, j=0;
    a = (float **)malloc(DIM * sizeof(float *));

    for (i = 0; i < DIM; i++) {
        a[i] = (float *)malloc(DIM * sizeof(float));
    }

    x = (float *)malloc(DIM * sizeof(float));
    y = (float *)malloc(DIM * sizeof(float));

    for (i = 0; i < DIM; i++) {
        x[i] = y[i] = 1.0f;
        for (j = 0; j < DIM; j++) {
            a[i][j] = 2.0f;
        }
    }
}

void allocate_atomic_memory(int rank,
        char **sbuf, char **tbuf, char **cbuf,
        char **win_base, size_t size, enum WINDOW type, MPI_Win *win)
{
    int page_size;
    int purehost = 0;
    int mem_on_dev = 0;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if ('H' == options.src && 'H' == options.dst) {
        purehost = 1;
    }

    if (rank == 0) {
        mem_on_dev = ('D' == options.src) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.dst) ? 1 : 0;
    }

    if (mem_on_dev) {
        CHECK(allocate_device_buffer(sbuf));
        set_device_memory(*sbuf, 'a', size);
        CHECK(allocate_device_buffer(win_base));
        set_device_memory(*win_base, 'b', size);
        CHECK(allocate_device_buffer(tbuf));
        set_device_memory(*tbuf, 'c', size);
        if (cbuf != NULL) {
            CHECK(allocate_device_buffer(cbuf));
            set_device_memory(*cbuf, 'a', size);
        }
    } else {
        CHECK(posix_memalign((void **)sbuf, page_size, size));
        memset(*sbuf, 'a', size);
        if (type != WIN_ALLOCATE) {
            CHECK(posix_memalign((void **)win_base, page_size, size));
            memset(*win_base, 'b', size);
        }
        CHECK(posix_memalign((void **)tbuf, page_size, size));
        memset(*tbuf, 'c', size);
        if (cbuf != NULL) {
            CHECK(posix_memalign((void **)cbuf, page_size, size));
            memset(*cbuf, 'a', size);
        }
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if (rank == 0) {
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            } else {
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (purehost) {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, (void *) win_base, win));
            } else {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

void free_atomic_memory (void *sbuf, void *win_baseptr, void *tbuf, void *cbuf, enum WINDOW win_type, MPI_Win win, int rank)
{
    int mem_on_dev = 0;
    MPI_CHECK(MPI_Win_free(&win));

    if (rank == 0) {
        mem_on_dev = ('D' == options.src) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.dst) ? 1 : 0;
    }

    if (mem_on_dev) {
        free_device_buffer(sbuf);
        free_device_buffer(win_baseptr);
        free_device_buffer(tbuf);
        if (cbuf != NULL) {
            free_device_buffer(cbuf);
        }
    } else {
        free(sbuf);
        if (win_type != WIN_ALLOCATE) {
            free(win_baseptr);
        }
        free(tbuf);
        if (cbuf != NULL) {
            free(cbuf);
        }
    }
}

void init_arrays(double target_time)
{

    if (DEBUG) {
        fprintf(stderr, "called init_arrays with target_time = %f \n",
                (target_time * 1e6));
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU || options.target == BOTH) {
    /* Setting size of arrays for Dummy Compute */
    int N = options.device_array_size;

    /* Device Arrays for Dummy Compute */
    allocate_device_arrays(N);

    double t1 = 0.0, t2 = 0.0;

    while (1) {
        t1 = MPI_Wtime();

        if (options.target == GPU || options.target == BOTH) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            call_kernel(A, d_x, d_y, N, &stream);

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        t2 = MPI_Wtime();
        if ((t2-t1) < target_time)
        {
            N += 32;

            /* Now allocate arrays of size N */
            allocate_device_arrays(N);
        } else {
            break;
        }
    }

    /* we reach here with desired N so save it and pass it to options */
    options.device_array_size = N;
    if (DEBUG) {
        fprintf(stderr, "correct N = %d\n", N);
        }
    }
#endif

}

#ifdef _ENABLE_CUDA_KERNEL_
void allocate_device_arrays(int n)
{
    /* First free the old arrays */
    free_device_arrays();

    /* Allocate Device Arrays for Dummy Compute */
    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_x, 1.0f, n));
    CUDA_CHECK(cudaMemset(d_y, 2.0f, n));
    is_alloc = 1;
}
#endif
/* vi:set sw=4 sts=4 tw=80: */
